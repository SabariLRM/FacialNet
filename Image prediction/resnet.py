import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7) 

model.load_state_dict(torch.load('resnet18_fer2013_optimized.pth', map_location=torch.device('cpu')))
model.eval() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(image_path):
    img = Image.open(image_path)
    img_tensor = inference_transforms(img).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        
    plt.imshow(img)
    plt.title(f"Predicted Emotion: {class_names[preds[0]]}")
    plt.axis('off')
    plt.show()


predict_emotion('image copy.png')