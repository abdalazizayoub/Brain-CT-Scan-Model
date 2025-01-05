import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn.functional import softmax


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avg = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = torch.load(r"C:\Users\abdul\Desktop\ندقش\Dataset\Model.pth", map_location=torch.device('cpu'))
model.eval()

t = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Grayscale(3),
    torchvision.transforms.ConvertImageDtype(torch.float),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    torchvision.transforms.Resize((64, 64)),
])
st.title("Brain CT Scan Classifier")
file = st.file_uploader(label="Upload Brain CT Scan for classification", type=["jpg", "png"])

if file:

    x = Image.open(file).convert("RGB")
    x = t(x)
    x = x.unsqueeze(0)

    logits = model(x)
    prob = softmax(logits, dim=1)
    pred = torch.argmax(prob)

    if pred == 1:
        st.write("The scanned brain seems to have a Tumor!")
    else:
        st.write("The scanned brain seems Healthy.")
