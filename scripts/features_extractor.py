from ultralytics import YOLO
import cv2
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import Utils

POSE_MODEL_PATH = './models/sardine_pose_detector/sardine_front_pose_s_model_364.pt'
DATASET_FILE_PATH = './datasets/sardine_classifier_dataset/'
POSE_CONF = 0.85
DEBUG = True

poseModel = YOLO(POSE_MODEL_PATH)
profile_dataset = {
    'vectors': []
}
face_dataset = {
    'vectors': []
}

FISH = sys.argv[1]

VIDEO_PATH = f'./videos/Sardine_{FISH}.mp4'
video = cv2.VideoCapture(VIDEO_PATH)
n_frame = 0
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = video.read()
    if not ret:
        break

    poseResults = poseModel(frame, conf=POSE_CONF, verbose=False)
    
    # Process results list
    for result in poseResults:
        # Extract bounding boxes
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            
            # Extract keypoints if available
            keypoints_data = None
            if result.keypoints is not None:
                keypoints_data = result.keypoints.xy.cpu().numpy()  # Get keypoints in xy format
            
            # Process each detection
            for i in range(len(boxes)):
                # Convert bbox from xyxy to xywh format
                x1, y1, x2, y2 = boxes[i]
                x = int(x1)
                y = int(y1) 
                width = int(x2 - x1)
                height = int(y2 - y1)
                boxConfidence = result.boxes.conf.cpu()[i]
                
                # Extract keypoints for this detection
                keypoints_list = []
                if keypoints_data is not None and i < len(keypoints_data):
                    keypoints = keypoints_data[i]  # Get keypoints for this detection
                    
                    # Convert keypoints to list of tuples
                    for (kx, ky) in keypoints:
                        keypoints_list.append((int(kx), int(ky)))

                view = Utils.detect_view(keypoints_list)
                normalized_keypoint = Utils.normalize_keypoints(keypoints_list)

                if len(normalized_keypoint):
                    vectors = Utils.features_to_vectors(normalized_keypoint)
                    if view == 'PROFILE':
                        profile_dataset['vectors'].append(vectors.tolist())
                    else:
                        face_dataset['vectors'].append(vectors.tolist())

                    if DEBUG:
                        normalizedPointsImg = Utils.display_normalized_keypoints(normalized_keypoint)                    
                        cv2.imshow('normalizedPointsImg', normalizedPointsImg)

                        color = (255, 0, 0)
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                        
                        # Draw keypoints
                        Utils.draw_keypoints(frame, keypoints_list, color, radius=4)
                        
                        # Draw pose connections (optional)
                        Utils.draw_pose_connections(frame, keypoints_list, color, thickness=2)
                        
                        label_text = f"Sardine"
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Background rectangle for label
                        cv2.rectangle(frame, 
                                    (x, y - label_size[1] - 10), 
                                    (x + label_size[0] + 10, y), 
                                    color, -1)
                        
                        # Label text
                        cv2.putText(frame, label_text, (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Optional: Display confidence or additional info
                        info_text = f"Conf: {boxConfidence:.2f}"
                        cv2.putText(frame, info_text, (x, y + height + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if DEBUG:
            # Display the frame
            resized_frame = cv2.resize(frame, (int(3840*0.3), int(2160*0.3)))
            cv2.imshow('Pose Prediction', resized_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    n_frame += 1
    
    print(f'FRAME = {n_frame}/{total_frames}')

# Clean up
video.release()

# Save to JSON file
with open(f'{DATASET_FILE_PATH}profile_sardine_{FISH}.json', 'w') as f:
    json.dump(profile_dataset, f, indent=4)
with open(f'{DATASET_FILE_PATH}face_sardine_{FISH}.json', 'w') as f:
    json.dump(face_dataset, f, indent=4)