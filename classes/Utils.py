import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

KEYPOINT_NAMES = {
    'mouth': 0,
    'leftEye': 1,
    'rightEye': 2,
    'leftPectoralFinStart': 3,
    'leftPectoralFinEnd': 4,
    'rightPectoralFinStart': 5,
    'rightPectoralFinEnd': 6,
    'leftPelvicFinStart': 7,
    'leftPelvicFinEnd': 8,
    'rightPelvicFinStart': 9,
    'rightPelvicFinEnd': 10,
    'ventFinStart': 11,
    'ventFinEnd': 12,
    'caudalStart': 13,
    'caudalTopEnd': 14,
    'caudalBottomEnd': 15,
    'dorsalFinStart': 16,
    'dorsalFinEnd': 17,
    'dorsalFinTop': 18,
    'leftLateralLineStart': 19,
    'rightLateralLineStart': 20,
    'leftLateralLineMiddle': 21,
    'rightLateralLineMiddle': 22
}

KEYPOINT_CONNECTIONS = [
    ('leftEye', 'rightEye'),
    ('mouth', 'leftEye'),
    ('mouth', 'rightEye'),
    ('leftEye', 'leftLateralLineStart'),
    ('rightEye', 'rightLateralLineStart'),
    ('leftEye', 'leftPectoralFinStart'),
    ('rightEye', 'rightPectoralFinStart'),
    ('leftPectoralFinStart', 'leftPectoralFinEnd'),
    ('rightPectoralFinStart', 'rightPectoralFinEnd'),
    ('leftLateralLineStart', 'leftPelvicFinStart'),
    ('leftPelvicFinStart', 'leftPelvicFinEnd'),
    ('leftPectoralFinStart', 'rightPectoralFinStart'),
    ('rightLateralLineStart', 'rightPelvicFinStart'),
    ('rightPelvicFinStart', 'rightPelvicFinEnd'),
    ('leftPelvicFinStart', 'rightPelvicFinStart'),
    ('leftLateralLineStart', 'leftLateralLineMiddle'),
    ('leftLateralLineMiddle', 'caudalStart'),
    ('rightLateralLineStart', 'rightLateralLineMiddle'),
    ('rightLateralLineMiddle', 'caudalStart'),
    ('caudalStart', 'caudalTopEnd'),
    ('caudalStart', 'caudalBottomEnd'),
    ('leftLateralLineMiddle', 'ventFinStart'),
    ('rightLateralLineMiddle', 'ventFinStart'),
    ('ventFinStart', 'ventFinEnd'),
    ('leftLateralLineStart', 'dorsalFinStart'),
    ('rightLateralLineStart', 'dorsalFinStart'),
    ('dorsalFinStart', 'dorsalFinEnd'),
    ('dorsalFinStart', 'dorsalFinTop')
]

SARDINES_COLOR = [
    (255,0,0),
    (0,255,0),
    (0,0,255)
]

SARDINE_NAMES = {
    0: 'A',
    1: 'B',
    2: 'C'
}

def draw_keypoints(frame, keypoints, color=(0, 255, 0), radius=5):
    """Draw keypoints on the frame"""
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # Only draw valid keypoints
            cv2.circle(frame, (x, y), radius, color, -1)
            # Optional: Add keypoint labels
            cv2.putText(frame, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
def normalize_keypoints(keypoints):
    # Convert keypoints to numpy array
    if isinstance(keypoints, dict):
        kp_array = np.array([keypoints[name] for name in sorted(KEYPOINT_NAMES.keys(), 
                              key=lambda x: KEYPOINT_NAMES[x])])
    else:
        kp_array = np.array(keypoints)

    view = detect_view(kp_array)
    
    if view == 'PROFILE':
        return normalize_profile_view(kp_array)
    else:
        return normalize_face_view(kp_array)

def detect_view(keypoints, threshold=0.15):
    # Convert keypoints to numpy array
    if isinstance(keypoints, dict):
        keypoints = np.array([keypoints[name] for name in sorted(KEYPOINT_NAMES.keys(), 
                              key=lambda x: KEYPOINT_NAMES[x])])
    else:
        keypoints = np.array(keypoints)

    # Extraire les points
    mouth = keypoints[KEYPOINT_NAMES['mouth']]
    caudal_start = keypoints[KEYPOINT_NAMES['caudalStart']]
    left_eye = keypoints[KEYPOINT_NAMES['leftEye']]
    right_eye = keypoints[KEYPOINT_NAMES['rightEye']]
    left_pectoral = keypoints[KEYPOINT_NAMES['leftPectoralFinStart']]
    right_pectoral = keypoints[KEYPOINT_NAMES['rightPectoralFinStart']]

    # Calculer les distances
    body_length = np.linalg.norm(caudal_start - mouth)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    pectoral_distance = np.linalg.norm(left_pectoral - right_pectoral)

    # Décision basée sur les distances normalisées
    if (eye_distance / body_length > threshold) or (pectoral_distance / body_length > threshold):
        return 'FACE'
    else:
        return 'PROFILE'
        
def normalize_profile_view(keypoints):
    mouth = keypoints[KEYPOINT_NAMES["mouth"]]

    # 1. Translate to mouth origin (0, 0)
    translated = keypoints - mouth
    
    # 2. Rotate to align caudalStart with x-axis (y=0)
    dx = translated[KEYPOINT_NAMES["caudalStart"]][0]
    dy = translated[KEYPOINT_NAMES["caudalStart"]][1]
    angle = np.arctan2(dy, dx)  # Angle between current and x-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated = (translated @ rotation_matrix)

    # 3. Normalize by body length
    body_length = np.linalg.norm(rotated[KEYPOINT_NAMES["caudalStart"]])
    normalized = rotated / body_length

    # get fish direction
    direction = 'RIGHT'
    if mouth[0] < rotated[KEYPOINT_NAMES["caudalStart"]][0]:
        direction = 'LEFT'

    # Convert back to dictionary
    aligned_keypoints = {}
    for name, idx in KEYPOINT_NAMES.items():
        aligned_keypoints[name] = [normalized[idx, 0], normalized[idx, 1]]
        if direction == 'LEFT':
            aligned_keypoints[name][1] *= -1

    return aligned_keypoints

def normalize_face_view(keypoints):
    translated = keypoints - keypoints[KEYPOINT_NAMES["mouth"]]

    # Aligner les yeux sur l'axe x
    left_eye = translated[KEYPOINT_NAMES["leftEye"]]
    right_eye = translated[KEYPOINT_NAMES["rightEye"]]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.arctan2(dy, dx)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = translated @ rotation_matrix

    # Normaliser par la largeur de la tête
    head_width = np.linalg.norm(rotated[KEYPOINT_NAMES["rightEye"]] - rotated[KEYPOINT_NAMES["leftEye"]])
    normalized = rotated / head_width

    # Convertir en dictionnaire
    aligned_keypoints = {name: [-normalized[KEYPOINT_NAMES[name], 0], normalized[KEYPOINT_NAMES[name], 1]] for name in KEYPOINT_NAMES}
    return aligned_keypoints

def display_normalized_keypoints(normalized_keypoints, view='PROFILE'):
    """
    Display normalized keypoints.
    
    Args:
        normalized_keypoints: Dictionary of normalized keypoints
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot each keypoint
    for name, point in normalized_keypoints.items():
        ax.scatter(point[0], point[1], label=name)
    
    for start, end in KEYPOINT_CONNECTIONS:
        if start in normalized_keypoints and end in normalized_keypoints:
            xs = [normalized_keypoints[start][0], normalized_keypoints[end][0]]
            ys = [normalized_keypoints[start][1], normalized_keypoints[end][1]]
            ax.plot(xs, ys, 'b-', alpha=0.3)
    
    # Set equal aspect ratio and add grid
    ax.set_aspect('equal')
    ax.grid(True)

    if view == 'PROFILE':
        ax.set_xlim(-0.1, 1.5)
        ax.set_ylim(-0.75, 0.75)
    elif view == 'FACE':
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Sardine mask')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    
    # Convert matplotlib figure to OpenCV image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Convert buffer to numpy array
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    
    # Decode image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return img

def draw_pose_connections(frame, keypoints, color=(255, 255, 255), thickness=2):
    """Draw connections between keypoints"""
    for connection in KEYPOINT_CONNECTIONS:
        pt1_idx, pt2_idx = connection
        if (KEYPOINT_NAMES[pt1_idx] < len(keypoints) and KEYPOINT_NAMES[pt2_idx] < len(keypoints) and
            keypoints[KEYPOINT_NAMES[pt1_idx]][0] > 0 and keypoints[KEYPOINT_NAMES[pt1_idx]][1] > 0 and
            keypoints[KEYPOINT_NAMES[pt2_idx]][0] > 0 and keypoints[KEYPOINT_NAMES[pt2_idx]][1] > 0):
            
            pt1 = keypoints[KEYPOINT_NAMES[pt1_idx]]
            pt2 = keypoints[KEYPOINT_NAMES[pt2_idx]]
            cv2.line(frame, pt1, pt2, color, thickness)

def features_to_vectors(features):
        # Convert to feature vectors
        feature_vec = []
        for key, value in features.items():
            feature_vec.extend(value)  # Flatten [x,y] pairs
                
        return np.array(feature_vec)
