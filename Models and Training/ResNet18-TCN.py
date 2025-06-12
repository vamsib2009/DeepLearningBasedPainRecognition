#Code of ResNet18-TCN model

#run_loso.py runs LOSO validation to "validate" all 87 test subjects 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------- CNN Feature Extractor ----------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=6, pretrained=False):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove avgpool and fc

        # Optional regularization
        self.dropout = nn.Dropout(0.2) #0.25

    def forward(self, x):  # (B*T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B*T, 512)
        features = self.dropout(features)
        return features

# ---------- Temporal Block ----------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.45) #0.45

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.45) #0.45

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):  # (B, C, T)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Ensure same shape
        if out.shape[2] != res.shape[2]:
            min_length = min(out.shape[2], res.shape[2])
            out = out[:, :, :min_length]
            res = res[:, :, :min_length]

        assert out.shape == res.shape, f"Shape mismatch: out {out.shape}, res {res.shape}"
        return self.relu(out + res)

# ---------- TCN ----------
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation,
                              padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # (B, C, T)
        return self.network(x)

# ---------- Attention Pooling ----------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        weights = self.attn(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)  # softmax across T
        weighted = x * weights  # (B, T, C)
        return weighted.sum(dim=1)  # (B, C)

# ---------- CNN + TCN + Attention Model ----------
class CNN_TCN_Model(nn.Module):
    def __init__(self, sequence_length, num_classes):
        super(CNN_TCN_Model, self).__init__()
        self.sequence_length = sequence_length

        self.cnn = CNNFeatureExtractor(in_channels=6)

        self.tcn = TCN(input_size=512, num_channels=[512, 256, 128], kernel_size=3, dropout=0.3)

        self.attn_pool = AttentionPooling(128)

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, sequence_mhi, sequence_ofi):
        sequence_mhi = self.pad_with_black_frames(sequence_mhi)
        sequence_ofi = self.pad_with_black_frames(sequence_ofi)

        combined = torch.cat([sequence_mhi, sequence_ofi], dim=2)  # (B, T, 6, H, W)
        B, T, C, H, W = combined.shape
        combined = combined.view(B * T, C, H, W)

        cnn_features = self.cnn(combined)  # (B*T, 512)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, 512)
        cnn_features = cnn_features.permute(0, 2, 1)  # (B, 512, T)

        tcn_out = self.tcn(cnn_features)  # (B, 128, T)
        pooled = self.attn_pool(tcn_out)  # (B, 128)

        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        return self.fc2(x)  # (B, num_classes)

    def pad_with_black_frames(self, sequence, target_length=None):
        batch_size, seq_len, C, H, W = sequence.shape
        if target_length is None:
            target_length = self.sequence_length

        if seq_len < target_length:
            frames_to_add = target_length - seq_len
            black_frames = torch.zeros((batch_size, frames_to_add, C, H, W), dtype=sequence.dtype, device=sequence.device)
            sequence = torch.cat([sequence, black_frames], dim=1)
        else:
            sequence = sequence[:, :target_length]
        return sequence


import os
import cv2
import numpy as np
from natsort import natsorted
import random

def load_images_generator(folder_path, img_size):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            frames = []
            for img_file in natsorted(os.listdir(video_path)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(video_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    frames.append(img)
            if frames:
                video_identifier = video_folder
                yield np.array(frames), class_folder, video_identifier

def prepare_data_generator(ofi_path, landmark_path, img_size):
    ofi_data_dict = {video_id: (frames, label)
                     for frames, label, video_id in load_images_generator(ofi_path, img_size)}
    landmark_data_dict = {video_id: (frames, label)
                          for frames, label, video_id in load_images_generator(landmark_path, img_size)}

    for video_id, (ofi_frames, ofi_label) in ofi_data_dict.items():
        if video_id in landmark_data_dict:
            landmark_frames, landmark_label = landmark_data_dict[video_id]
            if ofi_label != landmark_label:
                print(f"Label mismatch for video ID {video_id}: OFI label {ofi_label}, Landmark label {landmark_label}. Skipping.")
                continue
            yield ofi_frames, landmark_frames, ofi_label, video_id
        else:
            print(f"Video ID {video_id} not found in landmark data. Skipping.")

# ==============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_subject', type=str, required=True, help='ID of test subject')
args = parser.parse_args()

print(f"Test Subject: {args.test_subject}")


ofi_base_path = ''
landmark_base_path = ''
img_size = (128, 128)
test_subject_id = args.test_subject

# ==============================================================

all_data = []
test_data = []

for ofi_frames, landmark_frames, label, video_id in prepare_data_generator(ofi_base_path, landmark_base_path, img_size):
    subject_id = video_id.split('_')[0].strip()
    if subject_id == test_subject_id:
        test_data.append((ofi_frames, landmark_frames, label))
    else:
        all_data.append((ofi_frames, landmark_frames, label))

# Shuffle and split 15% from training set for validation
random.seed(42)
random.shuffle(all_data)
val_split_idx = int(0.15 * len(all_data))
val_data = all_data[:val_split_idx]
train_data = all_data[val_split_idx:]

# ==============================================================

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

# ==============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter

# ---------------------- Normalization ----------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

import torch
import torchvision.transforms as transforms

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


# ---------------------- Data Augmentation (Optional) ----------------------
train_transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(p=0.05),               #     # Horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=3),                  #   # Small angle rotations
    transforms.RandomGrayscale(p=0.1),                         # Occasionally convert to grayscale
    transforms.ToTensor(),
    normalize,
    GaussianNoise(std=0.02)
])

# ---------------------- Dataset Class ----------------------
class VideoDataset(Dataset):
    def __init__(self, aligned_data, sequence_length, num_classes, img_size, transform=None):
        self.aligned_data = aligned_data
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.aligned_data)

    def __getitem__(self, idx):
        ofi_frames, landmark_frames, label = self.aligned_data[idx]

        ofi_frames = torch.tensor(ofi_frames / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        landmark_frames = torch.tensor(landmark_frames / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)

        ofi_frames = torch.stack([normalize(frame) for frame in ofi_frames])
        landmark_frames = torch.stack([normalize(frame) for frame in landmark_frames])

        ofi_frames = pad_frames(ofi_frames, self.sequence_length)
        landmark_frames = pad_frames(landmark_frames, self.sequence_length)

        encoded_label = 0 if label == 'PA4' else 1
        return (ofi_frames, landmark_frames), torch.tensor(encoded_label, dtype=torch.long)

# ---------------------- Pad Frames ----------------------
def pad_frames(sequence, target_length):
    current_length = sequence.shape[0]
    if current_length < target_length:
        C, H, W = sequence.shape[1:]
        padding = torch.zeros((target_length - current_length, C, H, W), dtype=sequence.dtype)
        sequence = torch.cat([sequence, padding], dim=0)
    elif current_length > target_length:
        sequence = sequence[:target_length]
    return sequence

# ---------------------- Collate ----------------------
def custom_collate(batch):
    ofi_batch, landmark_batch, label_batch = [], [], []
    for (ofi_frames, landmark_frames), label in batch:
        ofi_batch.append(ofi_frames)
        landmark_batch.append(landmark_frames)
        label_batch.append(label)
    return (torch.stack(ofi_batch), torch.stack(landmark_batch)), torch.tensor(label_batch, dtype=torch.long)

import math

# ---------------------- LR Scheduler ----------------------
def adjust_learning_rate(optimizer, epoch, initial_lr, decay_factor=0.1, decay_every=8):
    if epoch % decay_every == 0 and epoch > 0:
        new_lr = initial_lr * (decay_factor ** (epoch // decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate adjusted to: {new_lr:.6f}")


# ---------------------- Hyperparameters ----------------------
batch_size = 8
sequence_length = 32
num_classes = 2
img_size = (128, 128)
initial_lr = 0.0001
num_epochs = 30
weight_decay = 1e-4
early_stop_patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Label distribution in training data:", Counter([label for _, _, label in train_data]))

train_dataset = VideoDataset(train_data, sequence_length, num_classes, img_size)
val_dataset = VideoDataset(val_data, sequence_length, num_classes, img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = VideoDataset(test_data, sequence_length, num_classes, img_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- Model ----------------------
model = CNN_TCN_Model(sequence_length, num_classes).to(device)

# ---------------------- Loss, Optimizer ----------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

# ---------------------- Training Loop with Early Stopping ----------------------
best_val_loss = float('inf')
best_val_accuracy = 0.0
best_epoch = -1
patience_counter = 0

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, initial_lr)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for (ofi_frames, landmark_frames), labels in train_loader:
        ofi_frames = ofi_frames.to(device)
        landmark_frames = landmark_frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(ofi_frames, landmark_frames)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    avg_train_loss = total_loss / len(train_loader)

    # ---------------------- Validation ----------------------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for (ofi_frames, landmark_frames), labels in val_loader:
            ofi_frames = ofi_frames.to(device)
            landmark_frames = landmark_frames.to(device)
            labels = labels.to(device)

            outputs = model(ofi_frames, landmark_frames)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    # ---------------------- Save Best Model ----------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), f'best_model.pt_{test_subject_id}')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

print(f"Training Complete! Best Val Acc: {best_val_accuracy:.2f}% at Epoch {best_epoch}")

# ---------------------- Load Best Model & Evaluate on Test Set ----------------------
model.load_state_dict(torch.load(f'best_model.pt_{test_subject_id}'))
model.eval()

test_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for (ofi_frames, landmark_frames), labels in test_loader:
        ofi_frames = ofi_frames.to(device)
        landmark_frames = landmark_frames.to(device)
        labels = labels.to(device)

        outputs = model(ofi_frames, landmark_frames)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"\nTest Accuracy: {test_accuracy:.2f}% | Test Loss: {avg_test_loss:.4f}")
with open("output.txt", "a") as f:
    f.write(f"\n{test_subject_id} Test Accuracy: {test_accuracy:.2f}% | Test Loss: {avg_test_loss:.4f}\n")
