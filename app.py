import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# =============================
# LOAD MODELS (OFFLINE ONLY)
# =============================
@st.cache_resource
def load_models():
    # YOLO damage detection (local weights)
    damage_model = YOLO("best.pt")

    # ResNet-50 classification (NO pretrained download)
    resnet = resnet50(weights=None)
    resnet.fc = nn.Linear(2048, 6)  # number of damage classes
    resnet.load_state_dict(
        torch.load("resnet50_damage.pt", map_location="cpu")
    )
    resnet.eval()

    return damage_model, resnet

damage_model, resnet = load_models()

# =============================
# RESNET IMAGE TRANSFORM
# =============================
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# DAMAGE COST CONFIG
# =============================
damage_costs = {
    "dent": {"part": 300, "labor": 150},
    "scratch": {"part": 100, "labor": 50},
    "crack": {"part": 400, "labor": 180},
    "glass shatter": {"part": 600, "labor": 220},
    "lamp broken": {"part": 250, "labor": 100},
    "tire flat": {"part": 150, "labor": 80},
}

# =============================
# RESNET REFINEMENT LOGIC
# =============================
def refine_with_resnet(image_pil, boxes):
    preds = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image_pil.crop((x1, y1, x2, y2))
        tensor = resnet_transform(crop).unsqueeze(0)

        with torch.no_grad():
            logits = resnet(tensor)
            preds.append(logits)

    if preds:
        return torch.mean(torch.stack(preds), dim=0)

    return None

# =============================
# STREAMLIT UI
# =============================
st.title("🚗 Car Damage Detection & Repair Cost Estimator")

st.write(
    "Upload a damaged car image. "
    "The system localizes damage using YOLO and refines classification using ResNet-50."
)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
damage_file = st.file_uploader(
    "Upload damaged car image",
    type=["jpg", "jpeg", "png"]
)

if damage_file:
    damage_img = Image.open(io.BytesIO(damage_file.read())).convert("RGB")
    st.image(damage_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting damage..."):
        results = damage_model.predict(damage_img)
        r = results[0]

    st.image(
        r.plot(),
        caption="YOLO Damage Localization",
        use_container_width=True
    )

    if r.boxes is not None and len(r.boxes) > 0:
        refined_logits = refine_with_resnet(damage_img, r.boxes)
        refined_class = torch.argmax(refined_logits).item()
        label = damage_model.names[refined_class]

        st.subheader("🔍 Final Damage Classification")
        st.write(f"**{label.capitalize()}**")

        if label in damage_costs:
            part = damage_costs[label]["part"]
            labor = damage_costs[label]["labor"]
            total = part + labor

            st.subheader(f"💰 Estimated Repair Cost: ${total}")
            st.write(f"Part Cost: ${part}")
            st.write(f"Labor Cost: ${labor}")

            if st.button(f"Pay ${total}"):
                st.success(f"✅ Payment Successful! Amount Paid: ${total}")
                st.balloons()
        else:
            st.info("Detected damage does not have a predefined cost.")
    else:
        st.warning("⚠ No damages detected in the image.")
