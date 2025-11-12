# news_app.py - Streamlit Fake News Detection (compatible & Plotly-free)
import streamlit as st
import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

st.set_page_config(page_title="News Detection (Real/Fake)", layout="centered")

# ------------------ Helper functions ------------------
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def predict_text(model, vectorizer, text):
    """Return (label_str, confidence_float or None, prob_dict)"""
    x = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else [str(i) for i in range(len(probs))]
        prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        pred_index = int(np.argmax(probs))
        pred_label = str(classes[pred_index])
        confidence = float(probs[pred_index])
        return pred_label, confidence, prob_dict
    else:
        pred_label = model.predict(x)[0]
        return str(pred_label), None, {}

# ------------------ Load model & vectorizer ------------------
MODEL_FILES = ["random_forestmodel.pkl", "random_forestmodel (2).pkl", "/mnt/data/random_forestmodel.pkl"]
VECT_FILES = ["tfidfvectorizer.pkl", "tfidfvectorizer (1).pkl", "/mnt/data/tfidfvectorizer.pkl"]

model = None
vectorizer = None
model_path_used = None
vect_path_used = None

for p in MODEL_FILES:
    if os.path.exists(p):
        model = load_pickle(p)
        model_path_used = p
        break

for p in VECT_FILES:
    if os.path.exists(p):
        vectorizer = load_pickle(p)
        vect_path_used = p
        break

# ------------------ UI Layout ------------------
st.markdown("<h1 style='text-align:center; color: white;'>News Detection (Real / Fake)</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input column + example button column
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_area("Enter news text to check:", height=220, placeholder="Paste news text here...")
with col2:
    st.write("")  # spacing
    st.write("")
    if st.button("Try Example"):
        # use session_state to populate example without rerun
        st.session_state['example_text'] = (
            "Breaking: Central government announces a new policy expected to boost employment by 5% next year. "
            "Experts welcome the policy and say this will improve investor confidence."
        )

# If example exists in session_state, use it
if 'example_text' in st.session_state:
    user_input = st.session_state.pop('example_text')

analyze = st.button("Analyze this")

# Model info (collapsible)
with st.expander("Model Information", expanded=False):
    if model is not None and vectorizer is not None:
        st.write(f"**Model:** `{type(model).__name__}`")
        if hasattr(model, "classes_"):
            st.write(f"**Classes:** {model.classes_}")
        st.write("**Feature Extraction:** TF-IDF Vectorizer")
        if model_path_used:
            st.write(f"**Model file loaded from:** `{model_path_used}`")
        if vect_path_used:
            st.write(f"**Vectorizer loaded from:** `{vect_path_used}`")
    else:
        st.warning("Model or vectorizer not found. Place `random_forestmodel.pkl` and `tfidfvectorizer.pkl` in the same folder.")

# ------------------ Main prediction flow ------------------
if analyze:
    if not user_input or user_input.strip() == "":
        st.error("Please enter news text before analysis.")
    elif model is None or vectorizer is None:
        st.error("Model or vectorizer not found. Make sure the .pkl files are present in the app folder.")
    else:
        # Progress: simulated smooth progress
        progress_text = "Checking the authenticity of news... Please wait."
        with st.spinner(progress_text):
            prog = st.progress(0)
            for i in range(0, 101, 10):
                prog.progress(i)
                time.sleep(0.06)
            prog.empty()

        label, confidence, probs = predict_text(model, vectorizer, user_input)

        # Normalize label to display text and color
        lbl_low = str(label).lower()
        if lbl_low in ["fake", "0", "false"]:
            color = "#ff4b4b"
            icon = "❌"
            label_text = "FAKE"
        else:
            color = "#2ecc71"
            icon = "✅"
            label_text = "REAL"

        # Result tile
        st.markdown(
            f"<div style='background:{color};padding:12px;border-radius:8px'>"
            f"<h3 style='color:white;margin:0'>{icon} Result: This news appears to be <b>{label_text}</b></h3>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Show confidence bar (matplotlib)
        if confidence is not None:
            st.write(f"**Model Confidence:** {confidence*100:.2f}%")
            fig, ax = plt.subplots(figsize=(6, 0.9))
            ax.barh([0], [confidence], color=color)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Confidence")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

        # Show probability table and horizontal bar
        if probs:
            prob_df = pd.DataFrame(list(probs.items()), columns=["label", "prob"])
            prob_df = prob_df.sort_values("prob", ascending=True).reset_index(drop=True)

            st.write("**Prediction Probabilities:**")
            # formatted percent table
            display_df = prob_df.copy()
            display_df["prob"] = (display_df["prob"] * 100).map("{:.2f}%".format)
            st.table(display_df)

            # horizontal bar chart (matplotlib + seaborn aesthetics)
            fig2, ax2 = plt.subplots(figsize=(6, 1.8))
            bar_colors = [("#ff4b4b" if str(l).lower() in ["fake", "0", "false"] else "#2ecc71") for l in prob_df["label"]]
            ax2.barh(prob_df["label"], prob_df["prob"], color=bar_colors)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Probability")
            for i, v in enumerate(prob_df["prob"]):
                ax2.text(v + 0.01, i, f"{v*100:.2f}%", va="center")
            for spine in ax2.spines.values():
                spine.set_visible(False)
            st.pyplot(fig2)

        # Download result
        result_text = f"Input:\n{user_input}\n\nPrediction: {label_text}\nConfidence: {confidence}\n\nProbabilities: {probs}"
        st.download_button("Download Result", result_text.encode("utf-8"), file_name="prediction_result.txt")

st.markdown("---")

# ------------------ Footer Section ------------------
st.markdown(
    """
    <div style='text-align: center; padding: 15px; background-color: #111111; border-radius: 10px;'>
        <p style='color: #cccccc; font-size: 15px; margin: 0;'>
            <b>Developed by:</b> Mrunali & Tejashree &nbsp;|&nbsp; Department of Computer Science, Modern College, Pune
        </p>
        <p style='color: #888888; font-size: 14px; margin-top: 4px;'>
            <i>Powered by Machine Learning and NLP (Python · Scikit-learn · TF-IDF)</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

