import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np
from deep_sort_reid import DeepSort  # Fast and accurate re-ID model
from deep_sort_reid.model import ReIDModel  # Person re-identification model

# Directory to store trained identities
TRAINED_PERSON_DIR = "trained_persons"
os.makedirs(TRAINED_PERSON_DIR, exist_ok=True)

def train_person_reid(video_path: str, model_path: str = "yolov8n.pt"):
    """Detects a person in a video, extracts frames, and trains a model for re-identification."""
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    deepsort = DeepSort("deep_sort_model.pth")  # Load DeepSORT re-ID model
    reid_model = ReIDModel("reid_model.pth")  # Load person re-ID model
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    person_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Run YOLO inference
        detections = []
        for result in results:
            for box in result.boxes:  # Iterate over detections
                if int(box.cls) == 0:  # Class 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = frame[y1:y2, x1:x2]
                    person_frames.append(person_crop)
                    detections.append([x1, y1, x2, y2, box.conf[0].item()])
        
        # Update DeepSORT tracker
        tracks = deepsort.update(np.array(detections), frame)
        
        frame_id += 1
    
    cap.release()
    
    if not person_frames:
        print("No person detected in the video.")
        return
    
    # Create a unique person ID
    person_id = len(os.listdir(TRAINED_PERSON_DIR)) + 1
    person_folder = os.path.join(TRAINED_PERSON_DIR, f"person_{person_id}")
    os.makedirs(person_folder, exist_ok=True)
    
    # Save frames for training
    for i, frame in enumerate(person_frames):
        cv2.imwrite(os.path.join(person_folder, f"frame_{i}.jpg"), frame)
    
    print(f"Saved {len(person_frames)} frames for Person {person_id} at {person_folder}.")
    
    print("Training person re-identification model using collected frames...")
    # Train Re-ID model using collected frames
    reid_model.train(person_folder)
    reid_model.save("reid_model.pth")
    
    print(f"Person {person_id} is now registered for future detections.")