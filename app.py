import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app
st.title("📧 Spam Message Classifier")
st.write("Enter a message below to check if it's Spam or Not Spam.")

# Input box
user_input = st.text_area("Message Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize and predict
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)

        if result[0] == 1:
            st.error("🚫 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam.")
