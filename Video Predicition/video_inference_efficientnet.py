import argparse
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
from collections import deque
import statistics
import time

# --- VIDEO SOURCE AND OPTIONS ---
# Defaults to your webcam. Pass a file path to test a video instead:
#   python video_inference_efficientnet.py test_video.mp4
# Add --denoise to clean each face with DnCNN before classifying (needs ../Denoising/dncnn_rafdb_finetuned.pth)
parser = argparse.ArgumentParser(description="Real-time facial emotion recognition with EfficientNet-B0")
parser.add_argument('source', nargs='?', default=0, help="video file path (default: webcam 0)")
parser.add_argument('--denoise', action='store_true', help="denoise each face with the fine-tuned DnCNN first")
args = parser.parse_args()

# 1. Setup Device (NVIDIA GPU -> Apple Silicon GPU -> CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Recreate EfficientNet-B0 Architecture
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 7)
)

# 3. Load Weights
model.load_state_dict(torch.load('efficientnet_b0_rafdb_final.pth', map_location=device))
model.to(device)
model.eval()


class DnCNN(nn.Module):
    """DnCNN denoiser from Denoising/denoising.ipynb: predicts the noise map and subtracts it."""

    def __init__(self, channels=3, depth=20, features=64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, noisy):
        return noisy - self.model(noisy)


# Optional denoiser
denoiser = None
if args.denoise:
    denoiser = DnCNN()
    denoiser.load_state_dict(torch.load('../Denoising/dncnn_rafdb_finetuned.pth', map_location=device))
    denoiser.to(device)
    denoiser.eval()

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

cap = cv2.VideoCapture(args.source)

# Variables to calculate FPS
prev_frame_time = 0
new_frame_time = 0

model_label = "EfficientNet+DnCNN" if denoiser is not None else "EfficientNet"
print(f"✅ {model_label} Video feed starting on {device}... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to grab frame.")
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
        # 5% Padding, like the DenseNet script: RAF-DB faces are tightly aligned, and
        # extra background around the face costs accuracy (20% dropped it sharply in testing)
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

        # Optional DnCNN denoising on the face crop
        if denoiser is not None:
            with torch.no_grad():
                face = transforms.functional.to_tensor(pil_img).unsqueeze(0).to(device)
                pil_img = transforms.functional.to_pil_image(denoiser(face).clamp(0, 1)[0].cpu())

        input_tensor = img_transforms(pil_img).unsqueeze(0).to(device)

        # EfficientNet Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            raw_emotion = class_names[preds[0]]
            conf_percent = confidence.item() * 100

        emotion_window.append(raw_emotion)
        smooth_emotion = statistics.mode(emotion_window)

        # Draw box and emotion text
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        text = f"{model_label}: {smooth_emotion} ({conf_percent:.1f}%)"
        cv2.putText(frame, text, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion AI - EfficientNet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
