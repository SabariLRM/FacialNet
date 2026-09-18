import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
from collections import deque
import statistics
import time # Imported for FPS calculation

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Recreate Architecture
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),           
    nn.Linear(num_ftrs, 7)       
)

# 3. Load Weights
model.load_state_dict(torch.load('resnet18_rafdb_final.pth', map_location=device))
model.to(device)
model.eval() 

# 4. Image Transformations
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# RAF-DB Label Mapping
class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

# 5. Setup Face Detector & Smoothing Queue
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_window = deque(maxlen=10) 

# --- 6. VIDEO SOURCE TOGGLE ---
# Set to 0 for your default webcam. 
# To test a video file, change it to the file path (e.g., 'test_video.mp4')
VIDEO_SOURCE = 0 

cap = cv2.VideoCapture(VIDEO_SOURCE)

# Variables to calculate FPS
prev_frame_time = 0
new_frame_time = 0

print("✅ Video feed starting... Press 'q' on your keyboard to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to grab frame.")
        break

    # --- CALCULATE FPS ---
    new_frame_time = time.time()
    # Avoid division by zero on the very first frame
    if new_frame_time > prev_frame_time:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time
    
    # Convert FPS to an integer for a cleaner display
    fps_display = int(fps)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 20% Padding to capture eyebrows and chin
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        start_y = max(0, y - pad_y)
        end_y = min(frame.shape[0], y + h + pad_y)
        start_x = max(0, x - pad_x)
        end_x = min(frame.shape[1], x + w + pad_x)

        roi_color = frame[start_y:end_y, start_x:end_x]
        
        if roi_color.size == 0:
            continue
            
        pil_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        input_tensor = img_transforms(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            raw_emotion = class_names[preds[0]]
            conf_percent = confidence.item() * 100
            
        emotion_window.append(raw_emotion)
        smooth_emotion = statistics.mode(emotion_window)

        # Draw box and emotion text
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        
        # Draw Emotion AND Confidence Percentage
        text = f"DenseNet: {smooth_emotion} ({conf_percent:.1f}%)"
        cv2.putText(frame, text, (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # --- DRAW FPS COUNTER ---
    # Display the FPS in the top-left corner (x=10, y=30) in yellow
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()