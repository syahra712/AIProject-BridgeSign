import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 38
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Show live feed for 2 seconds before countdown
    start_time = time.time()
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        cv2.putText(frame, 'Get Ready...', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    # Countdown timer before capturing starts
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f'Starting in {countdown}...', (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)  # 1 second delay

    # Start capturing frames
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, f'Capturing {counter+1}/{dataset_size}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        cv2.waitKey(50)  # Adjust capture speed

cap.release()
cv2.destroyAllWindows()
