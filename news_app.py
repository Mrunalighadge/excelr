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
# --- Example & Clear Output Buttons Section ---
import streamlit as st

# Helper function to clear input/output
def clear_output():
    """Clears stored user input and results from Streamlit session state."""
    if 'user_input' in st.session_state:
        st.session_state['user_input'] = ''
    if 'result' in st.session_state:
        del st.session_state['result']
    if 'confidence' in st.session_state:
        del st.session_state['confidence']
    if 'probs' in st.session_state:
        del st.session_state['probs']
    if 'clear_clicked' in st.session_state:
        del st.session_state['clear_clicked']

# Layout for the buttons
colA, colB = st.columns(2)

with colA:
    if st.button("✨ Try Example"):
        st.session_state['user_input'] = (
            "Breaking: Central government announces a new policy expected to boost employment by 5% next year. "
            "Experts welcome the policy and say this will improve investor confidence."
        )
        st.info("✅ Example news inserted — click 'Analyze this' to check!")
        # for newer Streamlit
        try:
            st.experimental_rerun()
        except Exception:
            pass

with colB:
    if st.button("🧹 Clear Output"):
        clear_output()
        st.info("🧾 Output cleared. You can enter new text now!")
        # Version-safe rerun
        try:
            st.experimental_rerun()
        except Exception:
            st.session_state['clear_clicked'] = True

    
    if st.button("Analyze this"):
        progress_text = "Checking the authencity of news... Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
                my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(0.1)
                my_bar.empty()
        
    if user_input and user_input.strip():
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
