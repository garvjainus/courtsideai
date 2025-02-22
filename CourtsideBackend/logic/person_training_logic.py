import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np
import shutil
from pathlib import Path


# Directory to store trained identities
TRAINED_PERSON_DIR = "trained_persons"
os.makedirs(TRAINED_PERSON_DIR, exist_ok=True)

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # Calculate the intersection area
    inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))

    # Calculate the union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)

    return inter_area / float(box1_area + box2_area - inter_area)

def train_person_tracking(video_path: str, model_path: str = "mode/yolov8n.pt"):
    """Detects persons in a video, extracts frames, and matches persons using IoU for tracking."""
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    person_frames = []  # List to store frames with detected persons
    tracked_boxes = []  # List of boxes for each detected person

    # List to keep track of the identities of detected persons
    person_id_map = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLO inference
        detections = []
        
        # Process detections
        for result in results:
            for box in result.boxes:  # Iterate over detections
                if int(box.cls) == 0:  # Class 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    detections.append([x1, y1, x2, y2, conf])

        # Match detections from the current frame with tracked boxes from previous frames
        new_tracked_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            matched = False
            for track_id, prev_box in tracked_boxes:
                if iou([x1, y1, x2, y2], prev_box) > 0.5:  # IoU threshold to match
                    new_tracked_boxes.append((track_id, [x1, y1, x2, y2]))
                    matched = True
                    break
            if not matched:
                # If no match found, create a new tracking ID
                new_track_id = len(person_id_map) + 1
                person_id_map[new_track_id] = (x1, y1, x2, y2)
                new_tracked_boxes.append((new_track_id, [x1, y1, x2, y2]))
                
            # Save the frame whenever a person is detected or tracked
            person_crop = frame[y1:y2, x1:x2]
            person_frames.append(person_crop)

        tracked_boxes = new_tracked_boxes

        # Display the frame with tracked people
        for track_id, bbox in tracked_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

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
    
    print("Person has been registered with frames saved for future detections.")
    
    # Prepare the dataset and train the YOLOv8 model for the person

import yaml

def process_data_and_train(directory: str, model_save_path: str, image_size: int = 640, batch_size: int = 16):
    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model (YOLOv8n is a small model)

    # # Prepare dataset paths
    # train_dir = Path(directory)
    # image_paths = []
    # annotation_paths = []
    # class_names = {}

    # # Iterate over each person's folder (each person represents a class)
    # for person_id, person_folder in enumerate(train_dir.iterdir(), start=-1):
    #     if person_folder.is_dir():
    #         # class_names[person_id] = f"person_{person_id}"

    #         # Iterate over images in the person's folder
    #         for image_file in person_folder.glob("*.jpg"):  # Assuming all images are .jpg
    #             image_paths.append(str(image_file))  # Convert PosixPath to string
                
    #             # Prepare the annotation file path
    #             annotation_file = image_file.with_suffix(".txt")
    #             annotation_paths.append(str(annotation_file))  # Convert PosixPath to string

    #             # Process the annotations and save the corresponding .txt files
    #             with open(annotation_file, 'w') as f:
    #                 # Assume the bounding box is the whole image for simplicity
    #                 image = cv2.imread(str(image_file))
    #                 height, width, _ = image.shape
    #                 # Define the bounding box for the person as (center_x, center_y, width, height) in normalized coordinates
    #                 center_x = width / 2
    #                 center_y = height / 2
    #                 norm_width = width
    #                 norm_height = height

    #                 # Write the annotation in the format: class_id center_x center_y width height
    #                 f.write(f"{person_id - 1} {center_x} {center_y} {norm_width} {norm_height}\n")
    
    # # Create a dictionary for the data configuration (to be used in YAML)
    # data = {
    #     'train': list(image_paths),  # Convert PosixPath objects to strings
    #     'val': list(image_paths),    # For simplicity, using the same data for validation
    #     'names': {0: "person_0"}
    # }

    # # # Save the data dictionary to a YAML file
    yaml_file_path = Path(directory) / 'data.yaml'
    # with open(yaml_file_path, 'w') as yaml_file:
    #     yaml.dump(data, yaml_file)
    # print("yaml file path", yaml_file_path)
    
    # Train the model using the generated YAML file
    model.train(
        data="trained_persons/data.yaml",  # Pass the path to the generated YAML file
        imgsz=image_size,
        batch=batch_size,
        epochs=10,  # Specify number of epochs
        name='person_detection'
    )

    model.export(format="onnx")
    model.export(format="coreml")
    model.export(format="tflite")

# train_person_tracking("emi.MOV")
process_data_and_train('trained_persons', 'trained_persons')