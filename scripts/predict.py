from ultralytics import YOLO
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import Utils, Classifier

POSE_MODEL_PATH = './models/sardine_pose_detector/sardine_front_pose_s_model_364.pt'
PROFILE_CLASSIFIER_PATH = './models/profile_sardine_classifier/profile_sardine_classifier.h5'
FACE_CLASSIFIER_PATH = './models/face_sardine_classifier/face_sardine_classifier.h5'
INPUT_VIDEO_PATH = './videos/Sardine_ALL.mp4'
OUTPUT_VIDEO_PATH = './videos/predict_dresult.mp4'

POSE_THRESHOLD = 0.8
PROFILE_CLASSIFIER_THRESHOLD = 0.90
FACE_CLASSIFIER_THRESHOLD = 0.90

poseModel = YOLO(POSE_MODEL_PATH)
profileSardineClassifier = Classifier(model_path=PROFILE_CLASSIFIER_PATH)
profileSardineClassifier.load_model()

faceSardineClassifier = Classifier(model_path=FACE_CLASSIFIER_PATH)
faceSardineClassifier.load_model()

video = cv2.VideoCapture(INPUT_VIDEO_PATH)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
n_frame = 0
video.set(cv2.CAP_PROP_POS_FRAMES, n_frame)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (XVID, MJPG, DIVX, etc.)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (3840, 2160), True)

while True:
    ret, frame = video.read()
    if not ret:
        break

    poseResults = poseModel(frame, conf=POSE_THRESHOLD)
    
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
                        predictedSardine, sardineConfidence, _ = profileSardineClassifier.predict(vectors)
                        if sardineConfidence < PROFILE_CLASSIFIER_THRESHOLD:
                            continue
                    else:
                        predictedSardine, sardineConfidence, _ = faceSardineClassifier.predict(vectors)
                        if sardineConfidence < PROFILE_CLASSIFIER_THRESHOLD:
                            continue

                    color = Utils.SARDINES_COLOR[predictedSardine]
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    
                    # Draw keypoints
                    Utils.draw_keypoints(frame, keypoints_list, color, radius=4)
                    
                    # Draw pose connections (optional)
                    Utils.draw_pose_connections(frame, keypoints_list, color, thickness=2)
                    
                    # Draw cluster label
                    label_text = f"Sardine {Utils.SARDINE_NAMES[predictedSardine]}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                    
                    # Background rectangle for label
                    cv2.rectangle(frame, 
                                (x, y - label_size[1] - 10), 
                                (x + label_size[0] + 10, y), 
                                color, -1)
                    
                    # Label text
                    cv2.putText(frame, label_text, (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    
                    # Optional: Display confidence or additional info
                    info_text = f"{(sardineConfidence*100):.2f}%"
                    cv2.putText(frame, info_text, (x, y + height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    # Display the frame
    resized_frame = cv2.resize(frame, (int(3840*0.35), int(2160*0.35)))
    cv2.imshow('Pose Prediction', resized_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)

    n_frame += 1

    print(f'FRAME = {n_frame}')
# Clean up
video.release()
out.release()