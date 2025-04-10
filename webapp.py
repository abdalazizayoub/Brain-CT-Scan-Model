import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn.functional import softmax




model = torch.load("models/resnet.pth", map_location=torch.device('cpu'),weights_only=False)
model.eval()

t = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
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
