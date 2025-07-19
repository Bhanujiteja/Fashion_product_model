import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import joblib

from multi_output_model import MultiOutputModel

# Load the label encoders
from sklearn.preprocessing import LabelEncoder
import torch

torch.serialization.add_safe_globals([LabelEncoder])

le_color = torch.load("le_color.pt")
le_type = torch.load("le_type.pt")
le_usage = torch.load("le_usage.pt")
le_gender = torch.load("le_gender.pt")


# Get number of classes for each task
num_baseColour = len(le_color.classes_)
num_masterCategory = len(le_type.classes_)
num_usage = len(le_usage.classes_)
num_gender = len(le_gender.classes_)

# Initialize model with correct output dimensions
model = MultiOutputModel(
    num_baseColour=num_baseColour,
    num_masterCategory=num_masterCategory,
    num_gender=num_gender,
    num_usage=num_usage
)

model.load_state_dict(torch.load("fashion_model_full.pth", map_location=torch.device("cpu")))


# Load encoders
from sklearn.preprocessing import LabelEncoder
from torch.serialization import safe_globals

with safe_globals([LabelEncoder]):
    le_color = torch.load("le_color.pt", weights_only=False)
    le_type = torch.load("le_type.pt", weights_only=False)
    le_usage = torch.load("le_usage.pt", weights_only=False)
    le_gender = torch.load("le_gender.pt", weights_only=False)

# Set device
device = torch.device("cpu")
model.to(device)
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(model, image):  # change `image_path` to `image` here for clarity
    model.eval()
    img = image.convert('RGB')  # ✅ Use image directly without reopening
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)

    return {
        "Color": le_color.inverse_transform([outputs['baseColour'].argmax().item()])[0],
        "Type": le_type.inverse_transform([outputs['masterCategory'].argmax().item()])[0],
        "Usage": le_usage.inverse_transform([outputs['usage'].argmax().item()])[0],
        "Gender": le_gender.inverse_transform([outputs['gender'].argmax().item()])[0]
    }

# Streamlit UI
st.title("🧥 Fashion Product Classifier")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "png", "webp"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_image(model, image)
    st.subheader("Predicted Labels")
    st.json(prediction)
