import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Load YOLOv11n model
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained weights

model = load_model()

# Cost dictionary for detected damages
damage_costs = {
    "dent": {"part": 300, "labor": 150},
    "scratch": {"part": 100, "labor": 50},
    "crack": {"part": 400, "labor": 180},
    "glass shatter": {"part": 600, "labor": 220},
    "lamp broken": {"part": 250, "labor": 100},
    "tire flat": {"part": 150, "labor": 80},
}

# Streamlit UI
st.title("🚗 Car Damage Detection & Repair Cost Estimator")
st.write("Upload an image of a damaged car, and the app will detect damages and estimate repair cost.")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image from memory
    img = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run YOLO prediction
    results = model.predict(img)

    for r in results:
        # Display detection with bounding boxes
        res_plotted = r.plot()
        st.image(res_plotted, caption="Detected Damages", use_container_width=True)

        if len(r.boxes) > 0:
            # Find the highest probability detection
            best_idx = r.boxes.conf.argmax().item()
            best_class = int(r.boxes.cls[best_idx].item())
            best_conf = float(r.boxes.conf[best_idx].item())
            label = model.names[best_class]

            st.subheader("🔍 Most Likely Damage:")
            st.write(f"- **{label.capitalize()}** (Confidence: {best_conf:.2f})")

            if label in damage_costs:
                part_cost = damage_costs[label]["part"]
                labor_cost = damage_costs[label]["labor"]
                total_cost = part_cost + labor_cost

                st.subheader(f"💰 Total Repair Cost: ${total_cost}")
                st.write(f"Part: ${part_cost}, Labor: ${labor_cost}")

                if st.button(f"Pay ${total_cost}"):
                    st.success(f"✅ Payment Successful! Amount Paid: ${total_cost}")
                    st.balloons()
        else:
            st.warning("No damages detected.")
