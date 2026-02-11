import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Quantum AI | Sim-to-Real Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
}
.section-divider {
    border-top: 1px solid #333;
    margin-top: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Quantum AI – Sim-to-Real Noise Analyzer")
st.caption("Machine Learning Surrogates for Estimating Quantum Noise Degradation")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Quantum Configuration")

n_qubits = st.sidebar.slider("Number of Qubits", 2, 6, 3)
depth = st.sidebar.slider("Circuit Depth", 5, 30, 15)
p1 = st.sidebar.slider("1Q Error (p₁)", 0.01, 0.05, 0.02, step=0.005)
p2 = st.sidebar.slider("2Q Error (p₂)", 0.05, 0.20, 0.10, step=0.01)
shots = st.sidebar.selectbox("Shots", [256, 512, 1024], index=0)

run_eval = st.sidebar.button("Run Evaluation")

# --------------------------------------------------
# Background Section
# --------------------------------------------------
with st.expander("📘 Background: What does this system do?"):
    st.markdown(
        """
        ### Motivation

        Near-term quantum computers are **noisy**.  
        Even when we design a quantum circuit perfectly, real hardware introduces
        errors from imperfect gates, decoherence, and measurement noise.

        Understanding **how much noise changes the behavior of a quantum circuit**
        is essential for:
        - benchmarking quantum hardware
        - designing error-mitigation strategies
        - deciding when classical simulation is still reliable

        ---

        ### ⚛️ Sim-to-Real Gap in Quantum Computing

        In this project, we compare two versions of the same quantum circuit:

        - **Ideal simulation** – assumes perfect quantum operations  
        - **Noisy simulation** – includes realistic gate errors (NISQ regime)

        The difference between these two outcomes is called the  
        **sim-to-real gap**.

        We quantify this gap using **Kullback–Leibler (KL) divergence**, which measures
        how different two probability distributions are.

        > **Higher KL divergence ⇒ stronger noise impact**

        ---

        ### 🤖 Why Use Machine Learning Surrogates?

        Running noisy quantum simulations is **computationally expensive**,
        especially as circuit depth and qubit count increase.

        Instead of simulating every configuration from scratch, we train
        **machine-learning surrogate models** that learn the mapping:

        ```
        (number of qubits, circuit depth, noise levels)
            → expected noise-induced degradation
        ```

        Once trained, these surrogates provide **fast approximations** of
        quantum noise effects.

        ---

        ### 📊 What Do the Metrics Mean?

        This system reports **two different metrics**, each with a distinct role:

        #### 1️⃣ KL Divergence (Physics-Level Metric)
        - Measures how much the **noisy quantum system** deviates from the ideal one
        - Depends **only on the quantum simulation**
        - Same for all surrogate models

        #### 2️⃣ RMSE (Model-Level Metric)
        - Measures how accurately a surrogate predicts the sim-to-real gap
        - Used to **compare different surrogate models**
        - Lower RMSE means better predictive performance

        ---

        ### 🧠 How to Read the Results

        - **KL divergence** tells you *how noisy the quantum system is*
        - **RMSE** tells you *which surrogate model is more accurate*
        - The recommended model is the one with the **lowest RMSE**

        This separation ensures the results are
        **scientifically correct and interpretable**.
        """
    )

# --------------------------------------------------
# Evaluation Logic
# --------------------------------------------------
if run_eval:

    payload = {
        "n_qubits": n_qubits,
        "depth": depth,
        "p1": p1,
        "p2": p2,
        "shots": shots
    }

    with st.spinner("Running quantum simulation..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/evaluate",
                json=payload,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection failed: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(f"Backend error: {response.text}")
        st.stop()

    data = response.json()
    results_df = pd.DataFrame(data["results"])

    # --------------------------------------------------
    # Sort strictly in UI (authoritative ranking)
    # --------------------------------------------------
    results_df = results_df.sort_values("rmse").reset_index(drop=True)
    best_model = results_df.iloc[0]

    # --------------------------------------------------
    # Top Metrics Row
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    kl_value = results_df["kl_divergence"].iloc[0]

    col1.metric("KL Divergence", f"{kl_value:.4f}")
    col2.metric("Best Model", best_model["model"])
    col3.metric("Best RMSE", f"{best_model['rmse']:.4f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Table + Chart Layout
    # --------------------------------------------------
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Model Comparison")
        st.dataframe(
            results_df.style.format({
                "kl_divergence": "{:.4f}",
                "rmse": "{:.4f}"
            }),
            use_container_width=True
        )

    with col_right:
        st.subheader("Surrogate Error (RMSE)")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(results_df["model"], results_df["rmse"])
        ax.set_ylabel("RMSE (lower is better)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

        st.pyplot(fig)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Recommendation Panel
    # --------------------------------------------------
    st.success(
        f"Recommended surrogate model: {best_model['model']}\n\n"
        f"This model achieves the lowest RMSE for the given noise configuration."
    )

    # --------------------------------------------------
    # Technical Summary
    # --------------------------------------------------
    with st.expander("Technical Summary"):
        st.write(f"""
        Qubits: {n_qubits}  
        Depth: {depth}  
        1Q Error (p₁): {p1}  
        2Q Error (p₂): {p2}  
        Shots: {shots}

        KL Divergence represents the system-level sim-to-real gap.
        RMSE represents surrogate model prediction accuracy.
        """)
