# -*- coding: utf-8 -*-
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
