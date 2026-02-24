# Fish Individual Recognition using Computer Vision & Deep Learning

<img src="images/predict_result.gif" alt="Detection example"/>

This project tackles a complex challenge: **individually recognizing nearly identical fish** (size, color, shape) in a moving aquarium by combining computer vision and deep learning.

## 🛠️ Solution Architecture

1. **Synthetic dataset generation** with Unity for fast annotation
2. **Detection and pose estimation** with YOLO
3. **Keypoint-to-mask transformation** (profile/face)
4. **Individual classification** based on subtle morphology

## 📦 Dataset

- **Annotations**: 23 anatomical keypoints per fish
- **Tools**: Procedural generation in Unity → **YOLOPoseExporter.cs**
- **Annotation examples**:

<img src="images/pose_annotation_1.jpg" alt="pose_annotation_1" width="50%"/><img src="images/pose_annotation_2.jpg" alt="pose_annotation_2" width="50%"/>
<img src="images/pose_annotation_3.jpg" alt="pose_annotation_3" width="50%"/><img src="images/pose_annotation_4.jpg" alt="pose_annotation_4" width="50%"/>

## 🧠 YOLO Model for Pose Estimation

### Configuration
- **Model**: YOLO11-pose (yolo11s-pose)
- **Keypoints**: 23 anatomical points

    <img src="images/keypoints_name.png" alt="Keypoints name" width="25%"/>

- **Input**: 640x640 pixels

### Learning curves (400 epochs)
<img src="images/pose_metrics.png" alt="Pose metrics"/>

### Pose predictions
<img src="images/pose_predictions.gif" alt="Pose predictions"/>

## ✨ Keypoint Normalization → Masks

Method that converts keypoints into masks to capture subtle morphological differences:

1. Keypoint alignment:
    - Profile → mouth - caudalStart
    - Face → leftEye - rightEye
2. Silhouette generation
3. Perspective normalization (fish length)

**Transformation examples**:
| Keypoints | Mask | Mask type |
|-----------|---------------|------------|
| <img src="images/keypoints_1.png" alt="keypoints_1"/> | <img src="images/mask_keypoints_1.png" alt="mask_keypoints_1" width="50%"/> | Profile |
| <img src="images/keypoints_2.png" alt="keypoints_2"/> | <img src="images/mask_keypoints_2.png" alt="mask_keypoints_2" width="50%"/> | Profile |
| <img src="images/keypoints_3.png" alt="keypoints_3"/> | <img src="images/mask_keypoints_3.png" alt="mask_keypoints_3" width="50%"/> | Face |
| <img src="images/keypoints_4.png" alt="keypoints_4"/> | <img src="images/mask_keypoints_4.png" alt="mask_keypoints_4" width="50%"/> | Face |


## 🎯 Individual Classification

### Architecture:

Global Parameters
| Parameter | Value |
|-----------|---------------|
| Input Shape | (46,) |
| Number of Classes | 3 (3 fish) |
| Optimizer | Adam with Cyclical Learning Rate |
| Initial learning rate | 0.001 |
| Beta1 | 0.9 |
| Beta2 | 0.999 |
| Epsilon | 1e-7 |
| Loss Function | sparse_categorical_crossentropy |

Architecture Diagram
```
InputLayer(shape=(46,))
│
├─ Dense(256, activation='relu', kernel_initializer='he_normal')
├─ BatchNormalization()
├─ Dropout(0.2)
│
├─ [Residual Block 1]
│   ├─ Dense(256, activation='relu', L2=1e-5) → BatchNorm → Dropout(0.3)
│   ├─ Dense(256, activation='relu', L2=1e-5) → BatchNorm
│   └─ Add() + Dropout(0.3)  # Residual connection
│
├─ Dense(128, activation='relu', L2=1e-5) → BatchNorm → Dropout(0.35)
│
├─ [Residual Block 2]
│   ├─ Dense(128, activation='relu', L2=1e-5) → BatchNorm → Dropout(0.35)
│   ├─ Dense(128, activation='relu', L2=1e-5) → BatchNorm
│   └─ Add() + Dropout(0.35)  # Residual connection
│
├─ Dense(64, activation='relu', L2=1e-5) → BatchNorm → Dropout(0.4)
│
├─ [Classification Head]
│   ├─ Dense(32, activation='relu') → BatchNorm → Dropout(0.5)
│   └─ Dense(num_classes, activation='softmax')
│
Model: "Functional"
```

Key Features
1. Residual Connections

    - Two residual blocks to avoid vanishing gradients.

    - Uses Add() to merge inputs/outputs.

2. Regularization

    - Progressive dropout (from 0.2 to 0.5).

    - L2 regularization (1e-5) on dense layers.

    - Batch normalization after each dense layer.

3. Optimization

    - Adam with standard parameters (beta1, beta2).

    - Prepared for Cyclical Learning Rate (to be implemented via callback).

### Face model
- **Metrics**:

<img src="images/face_trainingsetstats2.png" alt="Training metrics Face"/>

- **Learning curves (436 epochs)**:

<img src="images/face_classifier_training_metrics.png" alt="Confusion matrix Face"/>

- **Confusion matrix**:

<img src="images/face_trainingset.png" alt="Confusion matrix Face"/>
<img src="images/face_trainingsetstats.png" alt="Training metrics Face"/>

### Profile model
- **Metrics**:

<img src="images/profile_trainingsetstats2.png" alt="Training metrics Profile"/>

- **Learning curves (150 epochs)**:

<img src="images/profile_classifier_training_metrics.png" alt="Confusion matrix Face"/>

- **Confusion matrix**:

<img src="images/profile_trainingset.png" alt="Confusion matrix Profile"/>
<img src="images/profile_trainingsetstats.png" alt="Training metrics Profile"/>


## 🚀 Usage

```bash
# Installation
pip install -r requirements.txt

# Inference
python predict.py
