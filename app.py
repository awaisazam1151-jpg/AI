import streamlit as st
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pure AI & Neural Simulator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI & Neural Network Interactive Studio")
st.write("Explore artificial intelligence fundamentals—built with pure Python logic!")

# --- TABS FOR NAVIGATION ---
tab1, tab2, tab3 = st.tabs([
    "🧠 Artificial Neuron Simulator", 
    "🏥 AI Triage & Decision Engine", 
    "🎓 AI Specialist Knowledge Check"
])

# ==========================================
# TAB 1: ARTIFICIAL NEURON SIMULATOR
# ==========================================
with tab1:
    st.header("🧠 Single Artificial Neuron Simulator")
    st.write(
        "Every deep learning model is built from thousands of individual **Neurons**. "
        "A neuron receives inputs, multiplies them by weights, adds a bias, and passes the total through an **Activation Function**."
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Adjust Neuron Parameters")
        input1 = st.slider("Input 1 (x1):", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        weight1 = st.slider("Weight 1 (w1):", min_value=-5.0, max_value=5.0, value=1.5, step=0.1)
        
        st.divider()
        input2 = st.slider("Input 2 (x2):", min_value=-5.0, max_value=5.0, value=-1.0, step=0.1)
        weight2 = st.slider("Weight 2 (w2):", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        
        st.divider()
        bias = st.slider("Bias (b):", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
        activation_choice = st.selectbox("Select Activation Function:", ["ReLU", "Sigmoid", "Linear"])

    with col2:
        st.subheader("2. Real-Time Math & Output")
        
        # Weighted Sum Calculation: Z = (x1 * w1) + (x2 * w2) + bias
        weighted_sum = (input1 * weight1) + (input2 * weight2) + bias
        
        # Apply Selected Activation Function
        if activation_choice == "ReLU":
            output = max(0.0, weighted_sum) # ReLU drops negative values to 0
        elif activation_choice == "Sigmoid":
            output = 1 / (1 + math.exp(-weighted_sum)) # Sigmoid squashes output between 0 and 1
        else: # Linear
            output = weighted_sum

        # Display Metrics
        st.metric(label="Weighted Sum (Z)", value=f"{weighted_sum:.2f}")
        st.metric(label=f"Neuron Output after {activation_choice}", value=f"{output:.4f}")

        st.subheader("Visual Output Signal")
        if activation_choice == "Sigmoid":
            st.progress(float(output))
        else:
            # Normalize display for progress bar
            norm_val = min(max(output / 10.0, 0.0), 1.0)
            st.progress(float(norm_val))

        # Explanation Card
        st.info(f"**How it works:** `Z = ({input1} × {weight1}) + ({input2} × {weight2}) + {bias} = {weighted_sum:.2f}`")

# ==========================================
# TAB 2: AI TRIAGE & DECISION ENGINE
# ==========================================
with tab2:
    st.header("🏥 Autonomous AI Decision Engine")
    st.write("Before modern deep learning, AI systems operated as **Expert Systems**—using complex conditional logic to make critical decisions.")

    st.subheader("Enter System Parameters:")
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Patient Age:", min_value=1, max_value=120, value=30)
        systolic_bp = st.slider("Systolic Blood Pressure (mmHg):", 80, 200, 120)
    with c2:
        heart_rate = st.slider("Heart Rate (BPM):", 40, 180, 75)
        has_chest_pain = st.checkbox("Experiencing Chest Pain")
        has_fever = st.checkbox("Experiencing High Fever")

    if st.button("🚀 Execute AI Risk Assessment"):
        risk_score = 0
        factors = []

        if systolic_bp > 140 or systolic_bp < 90:
            risk_score += 3
            factors.append("Abnormal Blood Pressure reading detected.")
        if heart_rate > 100 or heart_rate < 50:
            risk_score += 2
            factors.append("Irregular Heart Rate detected.")
        if has_chest_pain:
            risk_score += 4
            factors.append("Critical Indicator: Severe Chest Pain.")
        if age > 60 and has_fever:
            risk_score += 2
            factors.append("Age-related infection risk factor.")

        st.divider()
        st.subheader("Assessment Results:")
        
        if risk_score >= 5:
            st.error(f"🚨 **URGENT ATTENTION REQUIRED** (Calculated Risk Score: {risk_score}/11)")
        elif risk_score >= 2:
            st.warning(f"⚠️ **MODERATE PRIORITY** (Calculated Risk Score: {risk_score}/11)")
        else:
            st.success(f"✅ **NORMAL / LOW RISK** (Calculated Risk Score: {risk_score}/11)")

        st.write("**Identified Risk Factors:**")
        if factors:
            for f in factors:
                st.write(f"- {f}")
        else:
            st.write("- All parameters are within normal standard thresholds.")

# ==========================================
# TAB 3: AI KNOWLEDGE CHECK
# ==========================================
with tab3:
    st.header("🎓 AI Foundations Quiz")
    st.write("Test your knowledge on key Computer Science and AI concepts!")

    q1 = st.radio(
        "1. What is the main purpose of an **Activation Function** in a neural network?",
        [
            "To turn off the computer when training is done",
            "To introduce non-linearity so the network can learn complex patterns",
            "To save the code directly to a database"
        ]
    )

    if st.button("Submit Answer"):
        if "introduce non-linearity" in q1:
            st.balloons()
            st.success("🎉 Correct! Without activation functions, a neural network is just a basic linear equation.")
        else:
            st.error("❌ Not quite. Activation functions give neural networks the superpower to solve complex, curved problems!")
