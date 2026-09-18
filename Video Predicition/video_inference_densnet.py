import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F # ADDED: To calculate exact percentage probabilities
from torchvision import models, transforms
from PIL import Image
from collections import deque
import statistics
import time

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Recreate DenseNet Architecture
model = models.densenet121()
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),           
    nn.Linear(num_ftrs, 7)       
)

# 3. Load Weights 
model.load_state_dict(torch.load('densenet121_rafdb_final.pth', map_location=device))
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

VIDEO_SOURCE = 0 
cap = cv2.VideoCapture(VIDEO_SOURCE)

prev_frame_time = 0
new_frame_time = 0

print("✅ DenseNet Video feed starting... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    if new_frame_time > prev_frame_time:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time
    fps_display = int(fps)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # --- FIX 1: REDUCE PADDING TO 5% ---
        # This prevents DenseNet from getting distracted by the background
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        
        start_y = max(0, y - pad_y)
        end_y = min(frame.shape[0], y + h + pad_y)
        start_x = max(0, x - pad_x)
        end_x = min(frame.shape[1], x + w + pad_x)

        roi_color = frame[start_y:end_y, start_x:end_x]
        
        if roi_color.size == 0:
            continue
            
        pil_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        input_tensor = img_transforms(pil_img).unsqueeze(0).to(device)

        # DenseNet Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # --- FIX 2: CALCULATE CONFIDENCE PERCENTAGE ---
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            raw_emotion = class_names[preds[0]]
            conf_percent = confidence.item() * 100 # Convert math probability to a readable %
            
        emotion_window.append(raw_emotion)
        smooth_emotion = statistics.mode(emotion_window)

        # Draw UI Bounding Box
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2) 
        
        # Draw Emotion AND Confidence Percentage
        text = f"DenseNet: {smooth_emotion} ({conf_percent:.1f}%)"
        cv2.putText(frame, text, (start_x, start_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion AI - DenseNet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()