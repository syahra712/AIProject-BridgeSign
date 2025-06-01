import os
import pickle
import mediapipe as mp
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set the path to the data directory
DATA_DIR = '/Users/admin/Desktop/American-Sign-language-Detection-System/data'
EXPECTED_FEATURES = 42  # Expected number of features (21 landmarks × 2 coordinates)

data = []
labels = []

# Iterate through the dataset directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if it's a file (like .DS_Store)
    if not os.path.isdir(dir_path):
        continue

    # Iterate through images in each directory
    for img_path in os.listdir(dir_path):
        data_aux = []

        x_ = []
        y_ = []

        # Construct full image path
        img_full_path = os.path.join(dir_path, img_path)

        # Read the image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Could not read image {img_path} in {dir_}. Skipping...")
            continue
        
        # Convert image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Check if the data is complete before appending
            if len(data_aux) == EXPECTED_FEATURES:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Skipped image {img_path} in {dir_}: incomplete data with {len(data_aux)} features.")

# Save the dataset as a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved. Total samples: {len(data)}")
