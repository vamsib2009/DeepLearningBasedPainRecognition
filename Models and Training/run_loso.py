# Code to run LOSO validation 

import os
import subprocess
import sys
print(sys.executable)
import os
import cv2
import numpy as np
from natsort import natsorted
import random

ofi_base_path = ''
landmark_base_path = ''
img_size = (128, 128)
print("Hi")

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

def extract_all_subjects(ofi_path, landmark_path):
    subject_ids = set()
    for _, _, _, video_id in prepare_data_generator(ofi_path, landmark_path, img_size):
        subject_id = video_id.split('_')[0].strip()
        subject_ids.add(subject_id)
    return sorted(list(subject_ids))

def main():
    subjects = extract_all_subjects(ofi_base_path, landmark_base_path)


    for test_subject in subjects:
        print(f"\n==== Running LOSO Fold with Test Subject: {test_subject} ====\n")

        # Run the r.py script with the current test_subject
        subprocess.run([
            sys.executable, "ResNet18-TCN.py",  # Use sys.executable to use the Python interpreter from the virtual environment
            "--test_subject", test_subject
        ])

if __name__ == "__main__":
    main()
