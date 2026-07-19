import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#0E76A8;'>
        🩺 Chest X-Ray Pneumonia Detection
    </h1>
    <h4 style='text-align:center;color:gray;'>
        DenseNet121 + PyTorch + Streamlit
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

st.write("Upload a Chest X-Ray image to predict whether it is **NORMAL** or **PNEUMONIA**.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Class Labels
# -----------------------------
classes = ["NORMAL", "PNEUMONIA"]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = models.densenet121(weights=None)

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(nn.Linear(num_features, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(512, 2))


    model.load_state_dict(
        torch.load("/content/drive/MyDrive/chest_xray_Project/chest_xray_densenet121.pth", map_location = device))
    model.to(device)
    model.eval()

    return model


try:
    model = load_model()

except Exception as e:
    st.error(f"❌ Error loading model:\n\n{e}")
    st.stop()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if predicted_class == "NORMAL":

        st.success(f"Prediction: **{predicted_class}**")

    else:

        st.error(f"Prediction: **{predicted_class}**")

    st.metric(
        label="Confidence Score",
        value=f"{confidence_score:.2f}%"
    )

    st.progress(min(int(confidence_score), 100))

    st.markdown("### Prediction Probabilities")

    st.write(f"**NORMAL:** {probabilities[0][0].item()*100:.2f}%")
    st.write(f"**PNEUMONIA:** {probabilities[0][1].item()*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center><b>Developed using PyTorch • DenseNet121 • Streamlit</b></center>",
    unsafe_allow_html=True
)
