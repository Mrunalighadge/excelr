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
