import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to train the Naive Bayes model and save it as a pickle file
def train_model(data, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    model = MultinomialNB()
    model.fit(X, labels)

    # Save the model as a pickle file
    with open('spam_ham_model.pkl', 'wb') as file:
        pickle.dump((model, vectorizer), file)

    return model, vectorizer

# Function to load the trained model from the pickle file
def load_model():
    with open('spam_ham_model.pkl', 'rb') as file:
        model, vectorizer = pickle.load(file)
    return model, vectorizer

# Function to predict spam or ham
def predict(model, vectorizer, message):
    X = vectorizer.transform([message])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
def main():
    st.title("📧 Spam or Ham Detection")
    st.markdown("---")
    st.subheader("Detect whether a message is spam or ham")
     # Add an image with a smaller size
    st.image("https://media.istockphoto.com/id/1492989285/photo/people-with-warning-notification-and-spam-message-icon-on-mobile-phone.webp?b=1&s=170667a&w=0&k=20&c=eGEqVyp-oenpLl-m_4yAiGapLz6j6gZgnYnZRt1-Z9E=", width=200)

    # Add a beautiful background
    st.markdown(
        """
        <style>
            body {
                background-image: url("https://source.unsplash.com/1600x900/?nature,water");
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load the spam image
    spam_image_path = "spam_image.jpg"

    # Check if the model pickle file exists
    if not os.path.exists('spam_ham_model.pkl'):
        # Load the spam.csv dataset
        data = pd.read_csv('spam.csv')
        # Assume 'email' is the column containing emails and 'label' is the column containing labels
        if 'email' in data.columns and 'label' in data.columns:
            model, vectorizer = train_model(data['email'], data['label'])
            st.success("✅ Model trained successfully!")
        else:
            st.error("❌ Please make sure the CSV file contains 'email' and 'label' columns.")
    else:
        # Load the model from the pickle file
        model, vectorizer = load_model()

    # Predefined spam email message
    default_spam_message = "Congratulations! You've won a free vacation. Click here to claim your prize."

    # User input
    message = st.text_input("Enter a message", default_spam_message)

    # Predict
    if st.button("Predict"):
        if message.strip() == "":
            st.warning("⚠️ Please enter a message.")
        else:
            prediction = predict(model, vectorizer, message)
            if prediction == "spam":
                st.error("❌ This message is predicted to be spam.")
            else:
                st.success("✅ This message is predicted to be ham.")

    # How it works section
    st.markdown("---")
    st.subheader("How it works")
    st.write("""
    This application uses a Naive Bayes classifier to predict whether a given message is spam or ham.
    
    1. **Model Training:** The model is trained on a dataset containing email messages labeled as spam or ham.
    2. **Text Processing:** The text data is converted into numerical features using the Bag-of-Words approach.
    3. **Model Prediction:** When you input a message and click 'Predict', the model predicts whether it's spam or ham.
    4. **Result Display:** The prediction result is displayed along with a visual indicator (✅ for ham, ❌ for spam).
    """)
     st.write('Devlop By: Sanjaya Kumar Giri')
    st.write('[GitHub Repository](https://github.com/sanzzu-13)')
if __name__ == "__main__":
    main()
   
    
