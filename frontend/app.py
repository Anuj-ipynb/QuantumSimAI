import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(
    page_title="Quantum AI | Sim-to-Real Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Header
# ------------------------
st.markdown(
    """
    # 🧠 Quantum AI – Sim-to-Real Analyzer

    This tool evaluates **quantum noise effects** and compares
    **machine-learning surrogate models** for predicting
    sim-to-real degradation.

    ---
    """
)
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

# ------------------------
# Sidebar – Quantum Configuration
# ------------------------
st.sidebar.header("⚙️ Quantum Configuration")

n_qubits = st.sidebar.slider(
    "Number of Qubits",
    min_value=2,
    max_value=6,
    value=3
)

depth = st.sidebar.slider(
    "Circuit Depth",
    min_value=5,
    max_value=30,
    value=15
)

p1 = st.sidebar.slider(
    "1-Qubit Gate Error (p₁)",
    min_value=0.01,
    max_value=0.05,
    value=0.01,
    step=0.005
)

p2 = st.sidebar.slider(
    "2-Qubit Gate Error (p₂)",
    min_value=0.05,
    max_value=0.20,
    value=0.12,
    step=0.01
)

shots = st.sidebar.selectbox(
    "Measurement Shots",
    [256, 512, 1024],
    index=0
)

st.sidebar.markdown("---")

run_eval = st.sidebar.button("🚀 Run Evaluation")

# ------------------------
# Run evaluation
# ------------------------
if run_eval:
    payload = {
        "n_qubits": n_qubits,
        "depth": depth,
        "p1": p1,
        "p2": p2,
        "shots": shots
    }

    with st.spinner("Running quantum simulation and surrogate evaluation…"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/evaluate",
                json=payload,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Backend connection failed: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(f"❌ Backend error ({response.status_code}): {response.text}")
        st.stop()

    data = response.json()
    results_df = pd.DataFrame(data["results"])

    # ------------------------
    # Global system metric (KL)
    # ------------------------
    st.subheader("📐 Sim-to-Real Gap (Quantum System)")

    kl_value = results_df["kl_divergence"].iloc[0]

    st.metric(
        label="KL Divergence (Ideal vs Noisy Quantum System)",
        value=f"{kl_value:.4f}",
        help="This measures how much the noisy quantum system deviates from the ideal one. "
             "It is a property of the quantum system, not the surrogate model."
    )

    st.markdown("---")

    # ------------------------
    # Model comparison (RMSE)
    # ------------------------
    st.subheader("📊 Surrogate Model Comparison")

    results_df = results_df.sort_values("rmse")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.dataframe(
            results_df.style.format(
                {
                    "kl_divergence": "{:.4f}",
                    "rmse": "{:.4f}",
                }
            ),
            use_container_width=True
        )

    with col2:
        best_model = results_df.iloc[0]
        st.metric(
            label="🏆 Best Surrogate Model",
            value=best_model["model"],
            delta=f"RMSE = {best_model['rmse']:.4f}"
        )

    # ------------------------
    # RMSE plot (model-dependent)
    # ------------------------
    st.subheader("📉 Surrogate Prediction Error")

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        results_df["model"],
        results_df["rmse"]
    )

    ax.set_ylabel("RMSE (lower is better)")
    ax.set_xlabel("Surrogate Model")
    ax.set_title("Model-wise Surrogate Prediction Error")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    st.pyplot(fig)

    # ------------------------
    # Recommendation
    # ------------------------
    st.subheader("🧠 System Recommendation")

    st.success(
        f"""
        **Recommended surrogate model:** `{data['recommendation']}`

        This model achieves the **lowest prediction error (RMSE)** when
        approximating the quantum sim-to-real gap under the specified
        noise configuration.
        """
    )

    # ------------------------
    # Technical summary
    # ------------------------
    with st.expander("🔍 Technical Summary (for reviewers)"):
        st.markdown(
            f"""
            **Quantum setup**
            - Qubits: {n_qubits}
            - Circuit depth: {depth}
            - 1Q error (p₁): {p1}
            - 2Q error (p₂): {p2}
            - Shots: {shots}

            **Metrics**
            - **KL divergence**: global sim-to-real gap (physics-level)
            - **RMSE**: surrogate prediction error (model-level)

            **Interpretation**
            - KL divergence quantifies noise severity.
            - RMSE quantifies surrogate accuracy.
            """
        )
