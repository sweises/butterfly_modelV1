import torch
from PIL import Image
from torchvision import transforms, datasets
from models.resnet50 import create_model
import os, random

# gleiche Normalisierung wie beim Training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Klassen automatisch aus Ordnerstruktur einlesen
train_dir = "data/train"  
dataset = datasets.ImageFolder(train_dir)
classes = dataset.classes  
print("Possible Species", classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell laden
model = create_model(num_classes=len(classes))
model.load_state_dict(torch.load("resnet50_butterflies.pth", map_location=device))
model = model.to(device)
model.eval()

# Zufälliges Bild aus test-Ordner auswählen
test_dir = "test"
files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not files:
    raise RuntimeError(f"Keine Bilder im Ordner {test_dir} gefunden!")
img_name = random.choice(files)
img_path = os.path.join(test_dir, img_name)

# Bild laden
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# Vorhersage
with torch.no_grad():
    outputs = model(x)
    pred = outputs.argmax(1).item()

print(f"Actual Test Species: {img_name}")
print("Detected Species:", classes[pred])
