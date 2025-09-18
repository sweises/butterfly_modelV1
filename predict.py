import torch
from PIL import Image
from torchvision import transforms
from models.resnet50 import create_model

# gleiche Normalisierung wie beim Training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Klassen manuell angeben (aus train_dataset.classes auslesen)
classes = ["Art1", "Art2", "Art3", "Art4", "Art5"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=len(classes))
model.load_state_dict(torch.load("resnet50_butterflies.pth", map_location=device))
model = model.to(device)
model.eval()

img = Image.open("testbild.jpg")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(x)
    pred = outputs.argmax(1).item()

print("Vorhersage:", classes[pred])
