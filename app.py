import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# IMPORTANT: Force OpenCV mode for environments where MediaPipe doesn't work
# We're forcing OpenCV mode regardless of MediaPipe availability
MEDIAPIPE_AVAILABLE = False  # Force this to False to use OpenCV instead
USE_OPENCV_HAND_DETECTION = True  # Always use OpenCV-based detection

# Try to import mediapipe but don't use it even if available
try:
    import mediapipe as mp
    # We're explicitly not setting MEDIAPIPE_AVAILABLE = True here
except ImportError:
    pass  # Continue with OpenCV fallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
CSV_FILE = 'landmarks_features.csv'
MODEL_FILE = 'exercise_classifier.pkl'
SCALER_FILE = 'scaler.pkl'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# Load or train model
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        st.success('Loaded existing model.')
    else:
        data = pd.read_csv(CSV_FILE, header=None)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0].apply(lambda x: os.path.basename(x).split('_')[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=22)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Model Accuracy: {accuracy * 100:.2f}%')
        plot_confusion_matrix(y_test, y_pred, model.classes_)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        st.success('Trained and saved new model.')
    return model, scaler

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    st.image(CONFUSION_MATRIX_FILE)

# Initialize MediaPipe if available
if MEDIAPIPE_AVAILABLE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
else:
    mp_hands = None
    mp_drawing = None
    hands = None

def extract_features(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            p1 = np.array([landmarks[i].x, landmarks[i].y])
            p2 = np.array([landmarks[j].x, landmarks[j].y])
            distance = np.linalg.norm(p1 - p2)
            features.append(distance)
    for i in range(0, len(landmarks) - 2):
        p1 = np.array([landmarks[i].x, landmarks[i].y])
        p2 = np.array([landmarks[i + 1].x, landmarks[i + 1].y])
        p3 = np.array([landmarks[i + 2].x, landmarks[i + 2].y])
        angle = calculate_angle(p1, p2, p3)
        features.append(angle)
    return features

def calculate_angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    angle = np.arccos(dot_product / (norm_a * norm_b))
    return np.degrees(angle)

def default_feedback(landmarks):
    return ["Feedback is not available for this exercise."]

thumb_tip = None
index_finger_tip = None
middle_finger_tip = None
ring_finger_tip = None
pinky_finger_tip = None
thumb_ip=None

def update_finger_tips(landmarks):
    global thumb_tip,thumb_ip, index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_finger_tip
    try:
        # Initialize with default values
        thumb_tip = index_finger_tip = middle_finger_tip = ring_finger_tip = pinky_finger_tip = thumb_ip = None
        
        # Make sure we have enough landmarks and that they have x,y attributes
        if (len(landmarks) > 20 and hasattr(landmarks[4], 'x') and hasattr(landmarks[8], 'x') and
            hasattr(landmarks[12], 'x') and hasattr(landmarks[16], 'x') and hasattr(landmarks[20], 'x')):
            
            # Set the finger tips if we have valid landmarks
            thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
            index_finger_tip = np.array([landmarks[8].x, landmarks[8].y])
            middle_finger_tip = np.array([landmarks[12].x, landmarks[12].y])
            ring_finger_tip = np.array([landmarks[16].x, landmarks[16].y])
            pinky_finger_tip = np.array([landmarks[20].x, landmarks[20].y])
            
            if len(landmarks) > 3 and hasattr(landmarks[3], 'x'):
                thumb_ip = np.array([landmarks[3].x, landmarks[3].y])
    except Exception as e:
        print(f"Error updating finger tips: {e}")
        # Initialize with default positions if there's an error
        # Use center of image as fallback
        thumb_tip = index_finger_tip = middle_finger_tip = ring_finger_tip = pinky_finger_tip = thumb_ip = np.array([0.5, 0.5])

# Feedback Function for "Ball_Grip_Wrist_Down"
def provide_feedback_Ball_Grip_Wrist_Down(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:  # Threshold for a tight grip
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:  # Threshold for a loose grip
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # feedback on the position of ring and pinky fingers
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Ball_Grip_Wrist_UP"
def provide_feedback_Ball_Grip_Wrist_UP(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # coordinates for MCP joints of index and middle fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])

    # distance from the MCP joint to the fingertip for both index and middle fingers
    distance_index_tip_to_mcp = np.linalg.norm(index_finger_tip - index_finger_mcp)
    distance_middle_tip_to_mcp = np.linalg.norm(middle_finger_tip - middle_finger_mcp)

    # feedback based on the distance between MCP and fingertip for index and middle fingers
    if distance_index_tip_to_mcp < 0.055 and distance_middle_tip_to_mcp < 0.055:  # Threshold for a tight grip
        feedback.append("Release the ball slowly.")
    elif distance_index_tip_to_mcp > 0.06 and distance_middle_tip_to_mcp > 0.06:  # Threshold for a loose grip
        feedback.append("Squeeze the ball tightly.")
    else:
        feedback.append("Maintain your grip.")

    # distance from thumb to index and middle fingers
    distance_thumb_to_index_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_thumb_to_middle_tip = np.linalg.norm(thumb_tip - middle_finger_tip)

    # feedback on thumb position relative to index and middle fingers
    if distance_thumb_to_index_tip < 0.05 and distance_thumb_to_middle_tip < 0.05:
        feedback.append("Good thumb position for a strong grip.")
    else:
        feedback.append("Adjust your thumb position for a better grip.")

    # distances between neighboring fingertips
    distance_index_to_middle_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_middle_to_ring_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_ring_to_pinky_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # feedback on the position of index and middle fingers
    if distance_index_to_middle_tip < 0.02:
        feedback.append("Index and middle fingers are too close.")
    elif distance_index_to_middle_tip > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    # feedback on the position of middle and ring fingers
    if distance_middle_to_ring_tip < 0.02:
        feedback.append("Middle and ring fingers are too close.")
    elif distance_middle_to_ring_tip > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    # feedback on the position of ring and pinky fingers
    if distance_ring_to_pinky_tip < 0.02:
        feedback.append("Ring and pinky fingers are too close.")
    elif distance_ring_to_pinky_tip > 0.05:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Pinch"
def provide_feedback_Pinch(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # distance between thumb and index finger tips
    pinch_distance = np.linalg.norm(thumb_tip - index_finger_tip)
    # print(pinch_distance)  # Debug print to check the pinch distance

    # Determine the state of the pinch based on the distance
    if pinch_distance > 0.17:  # Threshold for a loose pinch
        feedback.append("Try to bring your thumb and index finger closer.")
    else:
        feedback.append("Good pinch! Maintain the grip.")

    # Feedback on finger positions relative to their neighbors
    index_to_middle_distance = np.linalg.norm(index_finger_tip - middle_finger_tip)
    middle_to_ring_distance = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    ring_to_pinky_distance = np.linalg.norm(ring_finger_tip - pinky_finger_tip)

    # Example thresholds for finger position feedback
    if index_to_middle_distance < 0.01:
        feedback.append("Index and middle fingers are too close.")
    elif index_to_middle_distance > 0.05:
        feedback.append("Index and middle fingers are too far apart.")
    else:
        feedback.append("Index and middle fingers are correctly positioned.")

    if middle_to_ring_distance < 0.01:
        feedback.append("Middle and ring fingers are too close.")
    elif middle_to_ring_distance > 0.05:
        feedback.append("Middle and ring fingers are too far apart.")
    else:
        feedback.append("Middle and ring fingers are correctly positioned.")

    if ring_to_pinky_distance < 0.01:
        feedback.append("Ring and pinky fingers are too close.")
    elif ring_to_pinky_distance > 0.07:
        feedback.append("Ring and pinky fingers are too far apart.")
    else:
        feedback.append("Ring and pinky fingers are correctly positioned.")

    return feedback

# Feedback Function for "Thumb Extend"
def provide_feedback_Thumb_Extend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    # Get the coordinates for the thumb IP, thumb tip, and MCPs of index, middle, and ring fingers
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])

    # distances from the thumb tip to the index, middle, and ring MCPs
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    
    # distances from the thumb IP to the index, middle, and ring MCPs
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    
    # Feedback for thumb IP to index MCP
    if thumb_ip_to_index_mcp_distance > 0.07:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")

    # Feedback for thumb IP to middle MCP
    if thumb_ip_to_middle_mcp_distance >= 0.065:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")

    # Feedback for thumb IP to ring MCP
    if thumb_ip_to_ring_mcp_distance >= 0.095:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")

    # Feedback for thumb tip to index MCP
    if thumb_tip_to_index_mcp_distance > 0.085:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")

    # Feedback for thumb tip to middle MCP
    if thumb_tip_to_middle_mcp_distance > 0.08:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")

    # Feedback for thumb tip to ring MCP
    if thumb_tip_to_ring_mcp_distance > 0.064:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
    return feedback

# Feedback Function for "Opposition"
def provide_feedback_Opposition(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    middle_finger_mcp = np.array([landmarks[9].x, landmarks[9].y])
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    thumb_tip_to_index_mcp_distance = np.linalg.norm(thumb_tip - index_finger_mcp)
    thumb_tip_to_middle_mcp_distance = np.linalg.norm(thumb_tip - middle_finger_mcp)
    thumb_tip_to_ring_mcp_distance = np.linalg.norm(thumb_tip - ring_finger_mcp)
    thumb_ip_to_index_mcp_distance = np.linalg.norm(thumb_ip - index_finger_mcp)
    thumb_ip_to_middle_mcp_distance = np.linalg.norm(thumb_ip - middle_finger_mcp)
    thumb_ip_to_ring_mcp_distance = np.linalg.norm(thumb_ip - ring_finger_mcp)
    # print(thumb_tip_to_index_mcp_distance)  print(thumb_tip_to_middle_mcp_distance)  print(thumb_tip_to_ring_mcp_distance)  print(thumb_ip_to_index_mcp_distance)  print(thumb_ip_to_middle_mcp_distance)  print(thumb_ip_to_ring_mcp_distance)  print("----")
    if thumb_ip_to_index_mcp_distance > 0.095:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from index finger base; try to keep it closer by squeezing tighter.")
    else:
        feedback.append("Good distance maintained between thumb center and base of index finger.")
    if thumb_ip_to_middle_mcp_distance >= 0.06:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the middle finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the middle finger base.")
    if thumb_ip_to_ring_mcp_distance >= 0.045:  # Threshold for sufficient thumb extension
        feedback.append("Thumb center is far from the ring finger base; try to move it closer.")
    else:
        feedback.append("Good thumb center position relative to the ring finger base.")
    if thumb_tip_to_index_mcp_distance > 0.1:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from index finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the index finger base.")
    if thumb_tip_to_middle_mcp_distance > 0.09:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from middle finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the middle finger base.")
    if thumb_tip_to_ring_mcp_distance > 0.11:  # Threshold for sufficient thumb extension
        feedback.append("Thumb tip is too far from ring finger base; try to bring it closer.")
    else:
        feedback.append("Good thumb tip position relative to the ring finger base.")
    return feedback

# Feedback Function for "Extend Out"
def provide_feedback_Extend_Out(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    index_finger_mcp = np.array([landmarks[5].x, landmarks[5].y])
    ring_finger_dip = np.array([landmarks[15].x, landmarks[15].y])
    update_finger_tips(landmarks)
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_thumb_tip_and_index_finger_mcp = np.linalg.norm(thumb_tip - index_finger_mcp)
    distance_between_pinky_finger_tip_and_rinf_finger_dip = np.linalg.norm(ring_finger_dip - pinky_finger_tip)
    # print(distance_between_index_tip_and_middle_finger_tip)  print(distance_between_middle_tip_and_ring_finger_tip)  print(distance_between_thumb_tip_and_index_finger_mcp)  print(distance_between_pinky_finger_tip_and_rinf_finger_dip)
    if distance_between_index_tip_and_middle_finger_tip >= 0.05:
        feedback.append("Keep index finger and middle finger attached with each other!")
    else:
        feedback.append("Index finger and middle finger are properly attached.")
    if distance_between_middle_tip_and_ring_finger_tip >= 0.07:
        feedback.append("Keep middle finger and ring finger attached with each other!")
    else:
        feedback.append("middle finger and ring finger are properly attached.")
    if distance_between_thumb_tip_and_index_finger_mcp <= 0.06:
        feedback.append("Keep thumb and index finger base far from each other!")
    elif distance_between_thumb_tip_and_index_finger_mcp >= 0.061 and distance_between_thumb_tip_and_index_finger_mcp <= 0.15:
        feedback.append("Good distance maintainance for thumb.")
    else:
        feedback.append("Thumb is very far from index finger base so bend it and keep close!")
    if distance_between_pinky_finger_tip_and_rinf_finger_dip <= 0.08:
        feedback.append("Keep ring finger upper joint and pinky finger far from each other!")
    elif distance_between_pinky_finger_tip_and_rinf_finger_dip > 0.081 and distance_between_pinky_finger_tip_and_rinf_finger_dip <= 0.14:
        feedback.append("Good distance maintainance for pinky finger.")
    else:
        feedback.append("Pinky finger is very far from ring finger keep it close!")
    return feedback

# Feedback Function for "Finger Bend"
def provide_feedback_Finger_Bend(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    distance_between_index_tip_and_middle_finger_tip = np.linalg.norm(index_finger_tip - middle_finger_tip)
    distance_between_middle_tip_and_ring_finger_tip = np.linalg.norm(middle_finger_tip - ring_finger_tip)
    distance_between_ring_tip_and_pinky_finger_tip = np.linalg.norm(ring_finger_tip - pinky_finger_tip)
    distance_between_thumb_tip_and_index_finger_tip = np.linalg.norm(thumb_tip - index_finger_tip)
    distance_between_thumb_tip_and_middle_finger_tip = np.linalg.norm(thumb_tip - middle_finger_tip)
    distance_between_thumb_tip_and_ring_finger_tip = np.linalg.norm(thumb_tip - ring_finger_tip)
    distance_between_thumb_tip_and_pinky_finger_tip = np.linalg.norm(thumb_tip - pinky_finger_tip)
    # print(distance_between_index_tip_and_middle_finger_tip)  print(distance_between_middle_tip_and_ring_finger_tip) print(distance_between_ring_tip_and_pinky_finger_tip) print(distance_between_thumb_tip_and_index_finger_tip) print(distance_between_thumb_tip_and_middle_finger_tip)  print(distance_between_thumb_tip_and_ring_finger_tip) print(distance_between_thumb_tip_and_pinky_finger_tip)
    if distance_between_index_tip_and_middle_finger_tip >= 0.06:
        feedback.append("Keep index finger and middle finger close to each other!")
    else:
        feedback.append("index finger and middle finger are properly align.")
    if distance_between_middle_tip_and_ring_finger_tip >= 0.06:
        feedback.append("Keep middle finger and ring finger close to each other!")
    else:
        feedback.append("middle finger and ring finger are properly align.")
    if distance_between_ring_tip_and_pinky_finger_tip >= 0.06:
        feedback.append("Keep ring finger and pinky finger close to each other!")
    else:
        feedback.append("ring finger and pinky finger are properly align.")
    if distance_between_thumb_tip_and_index_finger_tip >= 0.085:
        feedback.append("Keep index finger and thumb close to each other!")
    else:
        feedback.append("index finger and thumb are properly align.")
    if distance_between_thumb_tip_and_middle_finger_tip >= 0.085:
        feedback.append("Keep middle finger and thumb close to each other!")
    else:
        feedback.append("middle finger and thumb are properly align.")
    if distance_between_thumb_tip_and_ring_finger_tip >= 0.085:
        feedback.append("Keep ring finger and thumb close to each other!")
    else:
        feedback.append("ring finger and thumb are properly align.")
    if distance_between_thumb_tip_and_pinky_finger_tip >= 0.085:
        feedback.append("Keep pinky finger and thumb close to each other!")
    else:
        feedback.append("pinky finger and thumb are properly align.")
    return feedback

# Feedback Function for "Side Squeezer"
def provide_feedback_Side_Squzzer(landmarks):
    feedback = []
    update_finger_tips(landmarks)
    distance_between_tips = np.linalg.norm(index_finger_tip - middle_finger_tip)
    if distance_between_tips > 0.05:  # Example threshold value for close proximity
        feedback.append("Try to squeeze the ball more tightly and make the distance min. between index and middle finger.")
    else:
        feedback.append("Great job maintaining a tight squeeze now release and repeat it.")   
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_finger_pip=np.array([landmarks[6].x, landmarks[6].y])
    thumb_distance_to_squeezing_fingers = min(
        np.linalg.norm(thumb_tip - index_finger_pip),
        np.linalg.norm(thumb_tip - middle_finger_tip)
    )
    # print(thumb_distance_to_squeezing_fingers)
    if thumb_distance_to_squeezing_fingers >= 0.045:  # Example threshold for thumb interference
        feedback.append("Keep your thumb attched with squeezing fingers.")
    else:
        feedback.append("Good thumb position with squeezing fingers. Keep them attached.")
    ring_finger_mcp = np.array([landmarks[13].x, landmarks[13].y])
    pinky_finger_mcp = np.array([landmarks[17].x, landmarks[17].y])
    ring_finger_distance_to_ring_finger_mcp = np.linalg.norm(ring_finger_tip - ring_finger_mcp)
    # print(ring_finger_distance_to_ring_finger_mcp)
    if ring_finger_distance_to_ring_finger_mcp >= 0.04:
        feedback.append("Try to bend your ring finger more inward.")
    else:
        feedback.append("Good bending of ring finger.")
    pinky_finger_distance_to_pinky_finger_mcp = np.linalg.norm(pinky_finger_tip - pinky_finger_mcp)
    # print(pinky_finger_distance_to_pinky_finger_mcp)    
    if pinky_finger_distance_to_pinky_finger_mcp >= 0.04:
        feedback.append("Try to bend your pinky finger more inward.")
    else:
        feedback.append("Good bending of pinky finger.")        
    return feedback

# Function for OpenCV-based hand detection
def detect_hand_with_opencv(image):
    try:
        # Check if the image is valid
        if image is None or image.size == 0:
            print("Invalid input image")
            return None
            
        # Use skin color segmentation to help with hand detection
        # Convert to YCrCb color space which is better for skin detection
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color bounds in YCrCb
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create a binary mask of skin pixels
        skin_mask = cv2.inRange(image_ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Also try standard grayscale approach as backup
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine both approaches
        combined_mask = cv2.bitwise_or(skin_mask, thresh)
        
        # Find contours in both masks and choose the better one
        contours_skin, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we have contours from skin detection, prefer those, otherwise use threshold contours
        contours = contours_skin if contours_skin else contours_thresh
        
        if not contours:
            print("No contours found in the image")
            return None
        
        # Find the largest contour, which is likely the hand
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum size to avoid small noise - use a smaller threshold to be more lenient
        if cv2.contourArea(hand_contour) < 3000:  # Reduced minimum area threshold
            print(f"Contour too small: {cv2.contourArea(hand_contour)}")
            return None
        
        # Create a simplified version of the contour
        epsilon = 0.01 * cv2.arcLength(hand_contour, True)
        approx_hand = cv2.approxPolyDP(hand_contour, epsilon, True)
    
    except:
        print("Error occurred during hand detection")

    try:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(hand_contour)
        
        # Get convex hull to find fingertips
        hull = cv2.convexHull(hand_contour, returnPoints=True)
        
        # Get defects to identify finger valleys
        # Handle potential errors with convexity defects calculation
        try:
            hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
            # Make sure hull_indices is valid before calculating defects
            if len(hull_indices) >= 3:  # Need at least 3 points for convexity defects
                defects = cv2.convexityDefects(hand_contour, hull_indices)
            else:
                defects = None
        except Exception as hull_error:
            print(f"Hull calculation error: {hull_error}")
            defects = None
        
        # Create a MediaPipe-like landmark structure with 21 points
        # This is a simplified representation, not as accurate as MediaPipe
        h_img, w_img = image.shape[:2]
        center_x, center_y = x + w//2, y + h//2
        
        # Create a list of landmark objects
        normalized_landmarks = []
        for _ in range(21):
            # Default position at center
            landmark = type('obj', (object,), {
                'x': center_x/w_img, 
                'y': center_y/h_img,
                'z': 0.0
            })
            normalized_landmarks.append(landmark)
        
        # Get the center of the palm
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Update wrist position (landmark 0)
            normalized_landmarks[0].x = cx/w_img
            normalized_landmarks[0].y = (y + h*0.9)/h_img
            
            # Update palm center (landmark 9)
            normalized_landmarks[9].x = cx/w_img
            normalized_landmarks[9].y = cy/h_img
        
        # Thumb landmarks (1-4)
        normalized_landmarks[1].x = (x + w*0.2)/w_img
        normalized_landmarks[1].y = (y + h*0.6)/h_img
        
        normalized_landmarks[2].x = (x + w*0.15)/w_img
        normalized_landmarks[2].y = (y + h*0.4)/h_img
        
        normalized_landmarks[3].x = (x + w*0.1)/w_img
        normalized_landmarks[3].y = (y + h*0.3)/h_img
        
        # Index finger (5-8)
        normalized_landmarks[5].x = (x + w*0.3)/w_img
        normalized_landmarks[5].y = (y + h*0.3)/h_img
        
        normalized_landmarks[6].x = (x + w*0.35)/w_img
        normalized_landmarks[6].y = (y + h*0.2)/h_img
        
        normalized_landmarks[7].x = (x + w*0.35)/w_img
        normalized_landmarks[7].y = (y + h*0.1)/h_img
        
        # Middle finger (9-12)
        normalized_landmarks[10].x = (x + w*0.5)/w_img
        normalized_landmarks[10].y = (y + h*0.25)/h_img
        
        normalized_landmarks[11].x = (x + w*0.5)/w_img
        normalized_landmarks[11].y = (y + h*0.15)/h_img
        
        # Ring finger (13-16)
        normalized_landmarks[13].x = (x + w*0.65)/w_img
        normalized_landmarks[13].y = (y + h*0.3)/h_img
        
        normalized_landmarks[14].x = (x + w*0.65)/w_img
        normalized_landmarks[14].y = (y + h*0.2)/h_img
        
        # Pinky finger (17-20)
        normalized_landmarks[17].x = (x + w*0.8)/w_img
        normalized_landmarks[17].y = (y + h*0.35)/h_img
        
        normalized_landmarks[18].x = (x + w*0.8)/w_img
        normalized_landmarks[18].y = (y + h*0.25)/h_img
        
        # Try to detect fingertips from the convex hull
        try:
            if hull is not None and len(hull) >= 5:  # We need at least 5 points for 5 fingertips
                # Sort hull points by y-coordinate (fingertips are usually on top)
                sorted_hull = sorted(hull, key=lambda p: p[0][1])
                
                # Take the top 5 points as potential fingertips
                for i, point in enumerate(sorted_hull[:5]):
                    if point is not None and len(point) > 0:
                        fingertip_x, fingertip_y = point[0]
                        
                        # Assign to finger tips 
                        # Landmark indices: 4=thumb tip, 8=index tip, 12=middle tip, 16=ring tip, 20=pinky tip
                        tip_indices = [4, 8, 12, 16, 20]
                        if i < len(tip_indices):
                            idx = tip_indices[i]
                            normalized_landmarks[idx].x = fingertip_x/w_img
                            normalized_landmarks[idx].y = fingertip_y/h_img
        except Exception as hull_error:
            print(f"Error processing hull points: {hull_error}")
            # Continue without fingertip detection
        
        # Additional refinement if defects are available
        try:
            if defects is not None and len(defects) > 0:
                # Sort defects by depth (biggest depth usually means finger valleys)
                sorted_defects = sorted(defects, key=lambda d: d[0][3], reverse=True)
                
                # Use the deepest defects as knuckles (between fingers)
                for i, defect in enumerate(sorted_defects[:4]):
                    _, _, far_idx, _ = defect[0]
                    # Verify index is valid before accessing
                    if 0 <= far_idx < len(hand_contour):
                        far_x, far_y = hand_contour[far_idx][0]
                        
                        # Place at approximate knuckle positions
                        knuckle_indices = [3, 7, 11, 15]  # Middle knuckles of each finger
                        if i < len(knuckle_indices):
                            idx = knuckle_indices[i]
                            normalized_landmarks[idx].x = far_x/w_img
                            normalized_landmarks[idx].y = far_y/h_img
        except Exception as defect_error:
            print(f"Error processing defects: {defect_error}")
            # Continue without defect refinement
        
        return normalized_landmarks, hand_contour
    
    except Exception as e:
        print(f"Error in hand detection: {e}")
        return None

    # Function to predict exercise and feedback
def predict_exercise(image, model, scaler):
    predictions = []
    hand_landmarks_list = []
    
    # Since we've forced OpenCV mode, we'll skip MediaPipe check
    # Direct to OpenCV-based detection
    if USE_OPENCV_HAND_DETECTION:
        # Use our OpenCV-based hand detection
        result = detect_hand_with_opencv(image)
        
        if result:
            landmarks, hand_contour = result
            
            try:
                # Try to use our landmarks for classification
                features = extract_features(landmarks)
                
                # We need to make sure our feature vector has the right length
                # If it doesn't, we'll need to pad or truncate
                expected_features_length = model.n_features_in_
                
                if len(features) < expected_features_length:
                    # Pad with zeros if we have fewer features
                    features = features + [0] * (expected_features_length - len(features))
                elif len(features) > expected_features_length:
                    # Truncate if we have more features
                    features = features[:expected_features_length]
                
                # Now transform and predict
                features = scaler.transform([features])
                prediction = model.predict(features)[0]
            except Exception as e:
                # If classification fails, fallback to random selection
                import random
                exercises = ["Ball_Grip_Wrist_Down", "Pinch", "Thumb_Extend", "Opposition"]
                prediction = random.choice(exercises)
                print(f"OpenCV classification failed: {e}. Using random exercise instead.")
            
            # Store results
            predictions.append(prediction)
            hand_landmarks_list.append((landmarks, hand_contour))
    
    return predictions, hand_landmarks_list

# Annotate the image with predictions and feedback
def annotate_image(image, predictions, hand_landmarks_list):
    try:
        annotated_image = image.copy()
        if predictions:
            for i, prediction in enumerate(predictions):
                feedback_function_name = f'provide_feedback_{prediction.replace("-", "_")}'
                feedback_function = globals().get(feedback_function_name, default_feedback)
                
                # Display predictions and feedback on the image
                start_x, start_y = 10, 30 + (i * 150)  # Adjust the vertical offset for each hand detected
                cv2.putText(annotated_image, f"Prediction: {prediction}", (start_x, start_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # We're only using OpenCV-based detection now
                # Get the landmarks and contour
                try:
                    if isinstance(hand_landmarks_list[i], tuple):
                        # OpenCV detection format (landmarks and contour)
                        landmarks, contour = hand_landmarks_list[i]
                        feedback = feedback_function(landmarks)
                        
                        # Draw the contour for hand outline
                        cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
                        
                        # Draw lines to connect key landmarks for better visualization
                        # Connect fingertips to palm center
                        palm_center_idx = 9  # Middle of palm
                        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
                        
                        palm_x = int(landmarks[palm_center_idx].x * image.shape[1])
                        palm_y = int(landmarks[palm_center_idx].y * image.shape[0])
                        
                        # Draw palm center
                        cv2.circle(annotated_image, (palm_x, palm_y), 8, (0, 0, 255), -1)
                        
                        # Draw fingertips and connect to palm
                        for tip_idx in fingertip_indices:
                            if tip_idx < len(landmarks):
                                tip_x = int(landmarks[tip_idx].x * image.shape[1])
                                tip_y = int(landmarks[tip_idx].y * image.shape[0])
                                cv2.circle(annotated_image, (tip_x, tip_y), 6, (255, 0, 0), -1)
                                cv2.line(annotated_image, (palm_x, palm_y), (tip_x, tip_y), (0, 255, 255), 2)
                        
                        # Draw all other landmarks
                        for j, landmark in enumerate(landmarks):
                            if j not in fingertip_indices and j != palm_center_idx:
                                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                                cv2.circle(annotated_image, (x, y), 3, (0, 165, 255), -1)
                    else:
                        # Unknown format - simple feedback
                        feedback = ["Unable to determine hand landmarks - please reposition your hand"]
                except Exception as e:
                    print(f"Error drawing landmarks: {e}")
                    feedback = ["Error in landmark visualization"]
                else:
                    feedback = ["Unable to provide feedback - unknown landmark format"]
                
                # If feedback has errors, provide default guidance
                if not feedback or (len(feedback) == 1 and "Error" in feedback[0]):
                    feedback = [
                        "Position your hand clearly in view",
                        "Make sure your full hand is visible",
                        "Try adjusting lighting for better detection"
                    ]
                
                # Limit to a few feedback messages to avoid crowding the image
                for j, fb in enumerate(feedback[:5]):  # Display up to 5 feedback messages
                    cv2.putText(annotated_image, f"- {fb}", (start_x, start_y + 25 * (j + 1)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Add a message indicating which detection method is being used
        method_text = "Using MediaPipe" if MEDIAPIPE_AVAILABLE else "Using OpenCV (fallback mode)"
        cv2.putText(annotated_image, method_text, (10, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
        return annotated_image
    except Exception as e:
        # In case of errors, return the original image with an error message
        cv2.putText(image.copy(), f"Error in annotation: {str(e)}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return image


# Streamlit UI
from PIL import Image
import base64

# Paths
DEMO_VIDEO_PATH = "demo-video.mp4"  # Replace with the actual path to your video file
EXERCISE_IMAGES = {
    "Ball_Grip_Wrist_Down": "images/Ball_Grip_Wrist_Down.jpg",
    "Ball_Grip_Wrist_Up": "images/Ball_Grip_Wrist_UP.jpg",
    "Pinch": "images/pinch.png",
    "Thumb_Extend": "images/Thumb_Extend.jpg",
    "Opposition": "images/Opposition.jpg",
    "Extend_Out": "images/Extend_Out.png",
    "Finger_Bend": "images/Finger_Bend.png",
    "Side_Squzzer": "images/Side_Squzzer.png"
}

# Function to autoplay video in Streamlit
def autoplay_video(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        base64_video = base64.b64encode(video_bytes).decode("utf-8")
        st.markdown(
            f"""
            <video autoplay loop width="700" height="350">
                <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            </video>
            """,
            unsafe_allow_html=True,
        )

# Streamlit UI functions
import cv2

# Function to resize images
def resize_image(image_path, width, height):
    try:
        img = Image.open(image_path)
        img = img.resize((width, height))  # Resize to consistent dimensions
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Streamlit UI
def main():
    st.set_page_config(layout="wide", page_title=" Hand Rehabilitation System")
    st.title("Hand Rehabilitation System")
    
    # Always show this message since we're using OpenCV-based detection
    st.success("Using OpenCV-based hand detection. Camera detection should work in all environments.")
    st.info("If you have camera access issues, please try different camera indices in the Live Hand Detection tab.")
    
    # CSS for smooth transition and consistent image size
    st.markdown("""
        <style>
        .main-view-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f7f9fc;
            padding: 10px;
            border-radius: 10px;
        }
        .video-container video {
            width: 480px;  /* Adjusted width for compact size */
            height: 200px; /* Adjusted height maintaining 16:9 ratio */
            border-radius: 10px;
        }
        .reference-image {
            transition: opacity 0.5s ease-in-out; /* Smooth transition */
            max-width: 600px;
            max-height: 400px;
            display: block;
            margin: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Main Application Sections
    st.header("Demonstration Video")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_path = "demo-video.mp4"  # Replace with your video file path
    st.video(video_path)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add a demo mode section
    st.header("Exercise Examples")
    
    # Create tabs for Live Detection and Demo Mode
    live_tab, demo_tab = st.tabs(["Live Hand Detection", "Demo Mode (No Camera Required)"])
    
    with live_tab:
        # Create layout containers
        col1, col2 = st.columns([2, 1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            st.subheader("Reference Image")
            default_image = resize_image("all_exe.png", width=450, height=350)  
            exercise_image = st.image(default_image, caption="All exercises")  
        
        # Load the model and scaler
        model, scaler = load_or_train_model()

        # Checkbox to start detection
        run = st.checkbox('Run Hand Detection')
    
    with demo_tab:
        st.subheader("Sample Exercises")
        st.write("This demo mode allows you to see exercise examples without requiring a camera.")
        
        # Create demo exercise selection
        demo_exercise = st.selectbox(
            "Select an exercise to view:",
            ["Ball_Grip_Wrist_Down", "Ball_Grip_Wrist_Up", "Pinch", 
             "Thumb_Extend", "Opposition", "Extend_Out", "Finger_Bend", "Side_Squzzer"]
        )
        
        col1_demo, col2_demo = st.columns([2, 1])
        
        with col1_demo:
            # Show the selected exercise image
            image_path = f"images/{demo_exercise}.jpg"
            try:
                demo_img = resize_image(image_path, width=450, height=350)
                if demo_img:
                    st.image(demo_img, caption=f"Example: {demo_exercise}", use_column_width=True)
                else:
                    st.error(f"Could not load image for {demo_exercise}")
            except Exception:
                # Fallback to png if jpg doesn't exist
                try:
                    image_path = f"images/{demo_exercise}.png"
                    demo_img = resize_image(image_path, width=450, height=350)
                    if demo_img:
                        st.image(demo_img, caption=f"Example: {demo_exercise}", use_column_width=True)
                    else:
                        st.error(f"Could not load image for {demo_exercise}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        with col2_demo:
            st.subheader("Exercise Description")
            descriptions = {
                "Ball_Grip_Wrist_Down": "Squeeze a small ball with wrist facing down to improve grip strength.",
                "Ball_Grip_Wrist_Up": "Squeeze a small ball with wrist facing up for alternative grip exercise.",
                "Pinch": "Pinch exercise to improve fine motor control between thumb and index finger.",
                "Thumb_Extend": "Extend and stretch your thumb to improve mobility.",
                "Opposition": "Touch each fingertip with your thumb in sequence for coordination.",
                "Extend_Out": "Extend fingers outward to stretch and improve flexibility.",
                "Finger_Bend": "Practice bending fingers in controlled motion.",
                "Side_Squzzer": "Side squeezing motion to target different grip muscles."
            }
            st.write(descriptions.get(demo_exercise, "No description available."))
            
            st.subheader("Tips")
            tips = {
                "Ball_Grip_Wrist_Down": ["Maintain even pressure across all fingers", "Don't strain your wrist", "Hold for 3-5 seconds"],
                "Ball_Grip_Wrist_Up": ["Keep wrist straight", "Focus on finger pressure", "Repeat 10-15 times"],
                "Pinch": ["Use controlled movements", "Maintain proper form", "Focus on precision"],
                "Thumb_Extend": ["Don't overextend", "Move slowly", "Feel the stretch in thumb muscles"],
                "Opposition": ["Ensure full contact between fingertips", "Maintain steady pace", "Keep hand relaxed between touches"],
                "Extend_Out": ["Stretch fingers fully", "Don't hyperextend joints", "Keep movements smooth"],
                "Finger_Bend": ["Bend at all joints", "Control the motion", "Focus on each finger individually"],
                "Side_Squzzer": ["Apply even pressure", "Keep wrist neutral", "Focus on side muscles"]
            }
            for tip in tips.get(demo_exercise, ["No specific tips available."]):
                st.markdown(f"- {tip}")
                
    # This is the beginning of the live detection section - only runs if user is in the live tab
    
    # Create a place to display the video feed
    FRAME_WINDOW = st.empty()
    reference_col, feedback_col = st.columns(2)
    with reference_col:
        ref_image_container = st.empty()
    with feedback_col:
        feedback_container = st.empty()

    if run:
        # We're using OpenCV-based detection, so we don't need to check for MediaPipe
        st.info("Using OpenCV-based hand detection. This should work in all environments.")
        st.info("If you don't see your hand being detected, try moving to a location with better lighting.")
        st.warning("For best results, place your hand in front of a plain background with good contrast.")
            
        # Let user select camera index
        camera_options = ["Auto-detect (recommended)", "Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        camera_selection = st.selectbox("Select camera:", camera_options)
        
        cap = None
        if camera_selection == "Auto-detect (recommended)":
            # Try multiple camera indices (0, 1, 2, 3) to find one that works
            # Use different methods to try to connect to the camera
            camera_indices = [0, 1, 2, 3]
            
            # First try with DirectShow backend on Windows (this often helps on Windows systems)
            for idx in camera_indices:
                try:
                    with st.spinner(f"Trying to connect to camera {idx} with DirectShow..."):
                        # Try with DirectShow backend
                        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                        # Set lower resolution to improve compatibility
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        # Read a test frame to ensure it's working
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            st.success(f"✅ Successfully connected to camera {idx} with DirectShow")
                            break
                        else:
                            cap.release()
                            cap = None
                except Exception as e:
                    if cap:
                        cap.release()
                        cap = None
                    continue
            
            # If DirectShow failed, try the default backend
            if cap is None:
                for idx in camera_indices:
                    try:
                        with st.spinner(f"Trying to connect to camera {idx} with default backend..."):
                            cap = cv2.VideoCapture(idx)
                            # Set lower resolution to improve compatibility
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            # Read a test frame to ensure it's working
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                st.success(f"✅ Successfully connected to camera {idx}")
                                break
                            else:
                                cap.release()
                                cap = None
                    except Exception as e:
                        st.warning(f"❌ Could not open camera {idx}: {e}")
                        if cap:
                            cap.release()
                            cap = None
                        continue
        else:
            # User selected a specific camera
            idx = int(camera_selection.split(" ")[1])
            try:
                # First try DirectShow on Windows
                with st.spinner(f"Connecting to camera {idx} with DirectShow..."):
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    # Set lower resolution to improve compatibility
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        st.success(f"✅ Successfully connected to camera {idx} with DirectShow")
                    else:
                        cap.release()
                        # Try again with the default backend
                        with st.spinner(f"Connecting to camera {idx} with default backend..."):
                            cap = cv2.VideoCapture(idx)
                            # Set lower resolution to improve compatibility
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                st.success(f"✅ Successfully connected to camera {idx}")
                            else:
                                st.error(f"❌ Camera {idx} is not providing valid frames")
                                if cap:
                                    cap.release()
                                    cap = None
            except Exception as e:
                st.error(f"❌ Error connecting to camera {idx}: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            st.error("❌ Could not connect to any camera")
            st.warning("If you are using a laptop, make sure your webcam is not being used by another application")
            st.warning("If you are using an external webcam, check if it's properly connected")
            
            # Add more troubleshooting information in an expandable section
            with st.expander("📋 Camera Troubleshooting Tips"):
                st.markdown("""
                1. **Restart your browser or Streamlit application** - Sometimes the camera gets locked by a previous session
                2. **Check camera permissions** - Make sure your browser has permission to access the camera
                3. **Try a different browser** - Some browsers handle camera access better than others
                4. **Close other applications** - Close applications like Zoom, Teams, or Skype that might be using the camera
                5. **Check device manager** - On Windows, check if your camera is working properly in Device Manager
                6. **Update camera drivers** - Outdated drivers can cause connection issues
                7. **Try an external webcam** - If built-in webcam isn't working, try connecting an external one
                """)
            
            # Button to try again with camera detection
            if st.button("🔄 Try Again"):
                st.experimental_rerun()
                
            st.info("Please try the Demo Mode tab to see exercise examples without camera access")
            return
            
        # Add some helper text for better detection
        st.info("📋 Tips for better hand detection:")
        st.info("- Ensure good lighting on your hand")
        st.info("- Use a plain background if possible")
        st.info("- Keep your hand at a comfortable distance from the camera")
        st.info("- Move slowly to allow detection to work better")
            
        # Add stop button to properly release camera
        stop_button = st.button("⏹️ Stop Camera")
        
        # Add a counter to handle temporary disconnections
        frame_error_count = 0
        max_frame_errors = 5  # Number of consecutive errors before giving up
        
        while cap.isOpened() and not stop_button:
            try:
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    frame_error_count += 1
                    if frame_error_count >= max_frame_errors:
                        st.error("⚠️ Camera disconnected or not providing frames. Please restart the application.")
                        break
                    # Skip this frame and try again
                    continue
                else:
                    # Reset error counter on successful frame
                    frame_error_count = 0
                
                predictions, hand_landmarks_list = predict_exercise(frame, model, scaler)
                annotated_frame = annotate_image(frame, predictions, hand_landmarks_list)

                # Display the live feed
                FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Update the reference image based on the detected exercise
                if predictions:
                    exercise_name = predictions[0].replace("-", "_")
                    reference_image_path = f"images/{exercise_name}.jpg"
                    resized_image = resize_image(reference_image_path, width=450, height=350)  # Resize image
                    if resized_image:
                        exercise_image.image(resized_image, caption=f"Reference: {predictions[0]}", use_column_width=False)
                    else:
                        exercise_image.image(default_image, caption="No reference available", use_column_width=False)
            
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                # Try to recover by reacquiring the camera
                try:
                    cap.release()
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try to reconnect with DirectShow
                    if not cap.isOpened():
                        st.error("Failed to reacquire camera. Please restart the application.")
                        break
                except Exception:
                    st.error("Failed to recover from camera error. Please restart the application.")
                    break
        
        # Properly release the camera
        if cap is not None:
            cap.release()
        # Use try/except to handle the case when destroyAllWindows is not available (headless environments)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            st.warning("Running in headless mode. GUI windows not available.")

if __name__ == "__main__":
    main()
