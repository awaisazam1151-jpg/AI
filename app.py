import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# --- PAGE SETUP ---
st.set_page_config(
    page_title="CyberVision & AI Studio",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ CyberVision & Interactive AI Studio")
st.write("Welcome to your local AI & Data Processing Hub! Explore computer vision, pixel matrices, and dynamic modeling below.")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎮 Studio Modules")
module = st.sidebar.radio(
    "Choose a Module:",
    ["📷 Computer Vision & Filters", "📊 Neural Heatmap Simulator", "🤖 Smart Rule Triage Engine"]
)

# ==========================================
# MODULE 1: COMPUTER VISION & FILTERS
# ==========================================
if module == "📷 Computer Vision & Filters":
    st.header("📷 Module 1: Computer Vision & Edge Extraction")
    st.write("Computer vision systems see images as numerical matrices. Upload an image or use a sample gradient to extract structural features!")

    uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        # Create a synthetic sample gradient image if no upload
        st.info("No image uploaded. Generating synthetic test matrix...")
        arr = np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (300, 1))
        image = Image.fromarray(arr).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Vision Transformation")
        filter_type = st.selectbox(
            "Select Processing Algorithm:",
            ["Grayscale Matrix", "Edge Detection (Sobel-style)", "Thermal Heatmap", "Gaussian Blur", "Contour Mapping"]
        )

        if filter_type == "Grayscale Matrix":
            processed = ImageOps.grayscale(image)
        elif filter_type == "Edge Detection (Sobel-style)":
            processed = image.convert("L").filter(ImageFilter.FIND_EDGES)
        elif filter_type == "Thermal Heatmap":
            gray = ImageOps.grayscale(image)
            processed = ImageOps.colorize(gray, black="blue", white="red")
        elif filter_type == "Gaussian Blur":
            processed = image.filter(ImageFilter.GaussianBlur(radius=5))
        elif filter_type == "Contour Mapping":
            processed = image.convert("L").filter(ImageFilter.CONTOUR)

        st.image(processed, use_container_width=True)

    st.divider()
    st.subheader("🔬 Pixel Matrix Inspection")
    img_array = np.array(image)
    st.write(f"**Image Dimensions (Height x Width x Channels):** `{img_array.shape}`")
    st.write(f"**Average Red Channel Intensity:** `{np.mean(img_array[:,:,0]):.2f}` / 255")

# ==========================================
# MODULE 2: NEURAL HEATMAP SIMULATOR
# ==========================================
elif module == "📊 Neural Heatmap Simulator":
    st.header("📊 Module 2: Multi-Variable Pattern Simulator")
    st.write("Simulate how neural network weights or heatmaps respond to changing user parameters in real time.")

    col_a, col_b = st.columns(2)
    with col_a:
        resolution = st.slider("Matrix Resolution (Grid Size)", min_value=10, max_value=100, value=50)
    with col_b:
        frequency = st.slider("Signal Wave Frequency", min_value=1, max_value=10, value=3)

    # Generate wave pattern matrix
    x = np.linspace(-np.pi, np.pi, resolution)
    y = np.linspace(-np.pi, np.pi, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(frequency * X) + np.cos(frequency * Y)

    st.subheader("Interactive Activation Surface")
    st.caption("This heatmap visualizes mathematical signals similar to how artificial neurons process activation values.")
    
    # Convert numerical matrix directly to a heatmap display
    st.image(
        Image.fromarray(np.uint8((Z - Z.min()) / (Z.max() - Z.min()) * 255)), 
        caption=f"Generated Matrix Shape: ({resolution}x{resolution})",
        width=400
    )

# ==========================================
# MODULE 3: SMART TRIAGE ENGINE
# ==========================================
elif module == "🤖 Smart Rule Triage Engine":
    st.header("🤖 Module 3: Autonomous Decision Tree Engine")
    st.write("Before deep neural networks existed, expert systems used rule-based decision graphs to solve complex problems.")

    st.subheader("Patient Triage Assessment")
    age = st.number_input("Patient Age:", min_value=1, max_value=120, value=25)
    heart_rate = st.slider("Heart Rate (BPM):", min_value=40, max_value=180, value=75)
    symptoms = st.multiselect("Active Symptoms:", ["Fever", "Shortness of Breath", "Cough", "Chest Pain", "Fatigue"])

    if st.button("⚡ Run Rule Engine"):
        score = 0
        reasons = []

        if heart_rate > 100 or heart_rate < 50:
            score += 2
            reasons.append("Abnormal Heart Rate detected.")
        if "Chest Pain" in symptoms:
            score += 3
            reasons.append("High-priority symptom: Chest Pain.")
        if "Shortness of Breath" in symptoms:
            score += 2
            reasons.append("Respiratory distress noted.")
        if age > 65 and "Fever" in symptoms:
            score += 1
            reasons.append("Age risk factor combined with fever.")

        st.divider()
        if score >= 4:
            st.error(f"🚨 **Triage Level: HIGH URGENCY** (Risk Score: {score}/7)")
        elif score >= 2:
            st.warning(f"⚠️ **Triage Level: MODERATE PRIORITY** (Risk Score: {score}/7)")
        else:
            st.success(f"✅ **Triage Level: LOW PRIORITY / NORMAL** (Risk Score: {score}/7)")

        st.write("**Assessment Breakdown:**")
        for reason in reasons:
            st.write(f"- {reason}")
