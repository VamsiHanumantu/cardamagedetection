import streamlit as st
from ultralytics import YOLO
from PIL import Image
import easyocr
import io
import numpy as np

# =============================
# MOCK DATABASE
# =============================
REGISTERED_PLATES = {
    "KA05MN1234",
    "TS09AB4321",
    "MH12DE1433",
    "HO9BY9726",
    "HR2GDK8337"
}

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    damage_model = YOLO("best.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return damage_model, ocr_reader

damage_model, ocr_reader = load_models()

# =============================
# DAMAGE COSTS
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
# HELPERS
# =============================
def normalize_plate(text):
    return text.upper().replace(" ", "").replace("-", "")

def extract_plate(image_pil):
    image_np = np.array(image_pil)
    results = ocr_reader.readtext(image_np)

    ocr_texts = [normalize_plate(text) for _, text, _ in results]

    for plate in REGISTERED_PLATES:
        for t in ocr_texts:
            if plate in t or t in plate:
                return plate, ocr_texts

    return None, ocr_texts

# =============================
# SESSION STATE
# =============================
if "plate_verified" not in st.session_state:
    st.session_state.plate_verified = False
if "plate_number" not in st.session_state:
    st.session_state.plate_number = None

# =============================
# UI
# =============================
st.title("🚗 Car Damage Detection System")

# -------------------------------------------------
# STEP 1: NUMBER PLATE IMAGE UPLOAD
# -------------------------------------------------
st.header("Step 1️⃣ Upload Number Plate Image")

plate_file = st.file_uploader(
    "Upload image with clear number plate",
    type=["jpg", "jpeg", "png"],
    key="plate_uploader"
)

if plate_file and not st.session_state.plate_verified:
    plate_img = Image.open(io.BytesIO(plate_file.read())).convert("RGB")
    st.image(plate_img, caption="Uploaded Number Plate Image", use_container_width=True)

    with st.spinner("🔍 Detecting number plate..."):
        plate_number, ocr_texts = extract_plate(plate_img)

    st.subheader("🔎 OCR Detected Text")
    st.write(ocr_texts)

    if not plate_number:
        st.error("❌ Invalid or unregistered number plate.")
        st.stop()

    st.success(f"✅ Number Plate Verified: {plate_number}")
    st.session_state.plate_verified = True
    st.session_state.plate_number = plate_number

# -------------------------------------------------
# STEP 2: DAMAGE IMAGE UPLOAD (ONLY IF VERIFIED)
# -------------------------------------------------
if st.session_state.plate_verified:
    st.header("Step 2️⃣ Upload Damaged Car Image")

    damage_file = st.file_uploader(
        "Upload damaged car image",
        type=["jpg", "jpeg", "png"],
        key="damage_uploader"
    )

    if damage_file:
        damage_img = Image.open(io.BytesIO(damage_file.read())).convert("RGB")
        st.image(damage_img, caption="Uploaded Damage Image", use_container_width=True)

        with st.spinner("🚗 Detecting damage..."):
            results = damage_model.predict(damage_img)

        r = results[0]
        st.image(r.plot(), caption="Detected Damage", use_container_width=True)

        if r.boxes is not None and len(r.boxes) > 0:
            best_idx = r.boxes.conf.argmax().item()
            label = damage_model.names[int(r.boxes.cls[best_idx].item())]
            conf = float(r.boxes.conf[best_idx].item())

            st.subheader("🔍 Most Likely Damage")
            st.write(f"**{label.capitalize()}** (Confidence: {conf:.2f})")

            if label in damage_costs:
                part = damage_costs[label]["part"]
                labor = damage_costs[label]["labor"]
                total = part + labor

                st.subheader(f"💰 Estimated Repair Cost: ${total}")
                st.write(f"Part: ${part}, Labor: ${labor}")

                if st.button(f"Pay ${total}"):
                    st.success(
                        f"✅ Payment Successful!\n\n"
                        f"Plate: {st.session_state.plate_number}\n"
                        f"Amount: ${total}"
                    )
                    st.balloons()
        else:
            st.warning("⚠ No damages detected.")
