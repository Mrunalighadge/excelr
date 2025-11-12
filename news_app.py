# Streamlit deployment code
import streamlit as st
import pickle
import time

def main():
    st.title("News Detection(Real/Fake)")
    with open('random_forestmodel.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidfvectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # User input
    user_input = st.text_area("Enter news text to check:")

    if st.button("Analyze this"):
        progress_text = "Checking the authencity of news... Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
                my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(0.1)
                my_bar.empty()
        
    if user_input.strip():
    # Transform user input
        input_vectorized = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(input_vectorized)[0]
        # Display result
        if prediction == 1:
            st.success("This news is Real.")
        else:
            st.error("This news is Fake.")
    else:
        st.warning("Please enter  text.")
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
        background-color: green;
        }
        </style>""",
        unsafe_allow_html=True,)        

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

def plot_probability_bar(probs):
    """
    probs: dict like {'fake':0.2,'real':0.8} or list/array of probs with model.classes_
    Safe: uses matplotlib only.
    """
    try:
        if isinstance(probs, dict):
            labels = list(probs.keys())
            values = list(probs.values())
        else:
            # fallback: try to convert sequence and use default labels
            values = list(probs)
            labels = [f"Class {i}" for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(6, 1.2))
        colors = ['#ff4b4b' if str(l).lower() in ['fake','0','false'] else '#2ecc71' for l in labels]
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 1)
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v*100:.2f}%", va='center')
        ax.set_xlabel("Probability")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        st.pyplot(fig)
    except Exception as e:
        # don't break the app
        st.write(f"Probability plot skipped ({e})")


# ------------------ FOOTER SECTION ------------------

st.markdown("""---""")
st.markdown(
    """
    <div style='text-align: center; padding: 15px; background-color: #111111; border-radius: 10px;'>
        <p style='color: #cccccc; font-size: 15px;'>
            <b>Developed by:</b> Mrunali & Tejashree <br>
            Department of Computer Science, Modern College, Pune
        </p>
        <p style='color: #888888; font-size: 14px; margin-top: -10px;'>
            <i>Powered by Machine Learning and NLP (Python · Scikit-learn · TF-IDF)</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
