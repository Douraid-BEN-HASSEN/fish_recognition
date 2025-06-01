from ultralytics import YOLO
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import Utils

POSE_MODEL_PATH = './models/sardine_pose_detector/sardine_front_pose_s_model_364.pt'

poseModel = YOLO(POSE_MODEL_PATH)

# img1
img1 = cv2.imread('./images/sardineHaute.jpg', cv2.IMREAD_COLOR)
poseResults = poseModel(img1, conf=0.85)

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
                color = (255, 0, 0)
                # Draw bounding box
                cv2.rectangle(img1, (x, y), (x + width, y + height), color, 2)
                
                # Draw keypoints
                Utils.draw_keypoints(img1, keypoints_list, color, radius=4)
                
                # Draw pose connections (optional)
                Utils.draw_pose_connections(img1, keypoints_list, color, thickness=2)
                
                # Draw cluster label
                label_text = f"Sardine"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(img1, 
                            (x, y - label_size[1] - 10), 
                            (x + label_size[0] + 10, y), 
                            color, -1)
                
                # Label text
                cv2.putText(img1, label_text, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Optional: Display confidence or additional info
                info_text = f"Conf: {boxConfidence:.2f}"
                cv2.putText(img1, info_text, (x, y + height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Display the frame
normalizedPointsImg = Utils.display_normalized_keypoints(normalized_keypoint, view=view)                    
cv2.imshow('normalizedPointsImg img1', normalizedPointsImg)

resized_frame = cv2.resize(img1, (int(3840*0.3), int(2160*0.3)))
cv2.imshow('Pose Prediction img1', resized_frame)

# img2
img2 = cv2.imread('./images/sardineLongue.jpg', cv2.IMREAD_COLOR)
poseResults = poseModel(img2, conf=0.85)

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
                color = (255, 0, 0)
                # Draw bounding box
                cv2.rectangle(img2, (x, y), (x + width, y + height), color, 2)
                
                # Draw keypoints
                Utils.draw_keypoints(img2, keypoints_list, color, radius=4)
                
                # Draw pose connections (optional)
                Utils.draw_pose_connections(img2, keypoints_list, color, thickness=2)
                
                # Draw cluster label
                label_text = f"Sardine"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(img2, 
                            (x, y - label_size[1] - 10), 
                            (x + label_size[0] + 10, y), 
                            color, -1)
                
                # Label text
                cv2.putText(img2, label_text, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Optional: Display confidence or additional info
                info_text = f"Conf: {boxConfidence:.2f}"
                cv2.putText(img2, info_text, (x, y + height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Display the frame
normalizedPointsImg = Utils.display_normalized_keypoints(normalized_keypoint, view=view)                    
cv2.imshow('normalizedPointsImg img2', normalizedPointsImg)

resized_frame = cv2.resize(img2, (int(3840*0.3), int(2160*0.3)))
cv2.imshow('Pose Prediction img2', resized_frame)

# img3
img3 = cv2.imread('./images/sardine_face_3.jpg', cv2.IMREAD_COLOR)
poseResults = poseModel(img3, conf=0.85)

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
                color = (255, 0, 0)
                # Draw bounding box
                cv2.rectangle(img3, (x, y), (x + width, y + height), color, 2)
                
                # Draw keypoints
                Utils.draw_keypoints(img3, keypoints_list, color, radius=4)
                
                # Draw pose connections (optional)
                Utils.draw_pose_connections(img3, keypoints_list, color, thickness=2)
                
                # Draw cluster label
                label_text = f"Sardine"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(img3, 
                            (x, y - label_size[1] - 10), 
                            (x + label_size[0] + 10, y), 
                            color, -1)
                
                # Label text
                cv2.putText(img3, label_text, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Optional: Display confidence or additional info
                info_text = f"Conf: {boxConfidence:.2f}"
                cv2.putText(img3, info_text, (x, y + height + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Display the frame
view = Utils.detect_view(normalized_keypoint)
normalizedPointsImg = Utils.display_normalized_keypoints(normalized_keypoint, view=view)                    
cv2.imshow('normalizedPointsImg img3', normalizedPointsImg)

resized_frame = cv2.resize(img3, (int(3840*0.3), int(2160*0.3)))
cv2.imshow('Pose Prediction img3', resized_frame)

cv2.waitKey(0)