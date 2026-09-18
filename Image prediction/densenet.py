import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121()
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 7)

state_dict = torch.load('densenet121_fer2013(1).pth', map_location=device)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = img_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[preds[0]]}")
    plt.axis('off')
    plt.show()

predict('image copy.png')