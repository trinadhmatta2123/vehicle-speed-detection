import cv2
import numpy as np

# Load the pre-trained vehicle detection model (e.g., Haarcascades)
vehicle_cascade = cv2.CascadeClassifier('cars1.xml')

# Open video capture
cap = cv2.VideoCapture("carsVid.mp4")

# Initialize variables for tracking
previous_frame = None
vehicle_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if previous_frame is not None:
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(previous_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Loop through detected vehicles
        for (x, y, w, h) in vehicles:
            # Calculate average flow within the vehicle's bounding box
            avg_flow = np.mean(flow[y:y+h, x:x+w], axis=(0, 1))
            vehicle_speed = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2) * fps
            
            # Draw bounding box and display speed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{vehicle_speed:.2f} km/h', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    previous_frame = gray.copy()

    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
