import streamlit as st
import pandas as pd
import joblib

# Load your saved model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# Streamlit App Title
st.title("Text Classification App")
st.write("This app classifies text into 'Spam' or 'Not Spam' categories. You can input text or upload a CSV file for bulk classification.")

# Section: Single Text Input
st.header("Single Text Classification")
user_input = st.text_area("Enter your text here:")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess the input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        confidence = model.predict_proba(input_vector).max() * 100

        # Display results
        st.write(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
        st.write(f"Confidence Score: {confidence:.2f}%")
    else:
        st.warning("Please enter some text for classification.")

# Section: Bulk File Classification
st.header("Bulk Classification from CSV")
uploaded_file = st.file_uploader("Upload a CSV file for bulk classification", type=["csv"])

if uploaded_file:
    try:
        # Attempt to read the file with the correct encoding
        df = pd.read_csv(uploaded_file, encoding='latin1')  # Use 'latin1' encoding for non-UTF-8 files
        if 'Text' in df.columns:
            input_vectors = vectorizer.transform(df['Text'])
            df['Prediction'] = model.predict(input_vectors)
            df['Confidence Score'] = model.predict_proba(input_vectors).max(axis=1) * 100

            # Display first few predictions
            st.write("First few predictions:")
            st.write(df.head())

            # Visualization: Prediction distribution
            st.subheader("Prediction Distribution")
            st.bar_chart(df['Prediction'].value_counts())

            # Confidence threshold filter
            st.subheader("Filter Results by Confidence Score")
            threshold = st.slider("Confidence Threshold", 0, 100, 50)
            filtered_df = df[df['Confidence Score'] >= threshold]
            st.write("Filtered Results:")
            st.write(filtered_df)

            # Download filtered predictions
            st.subheader("Download Filtered Predictions")
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Predictions",
                data=csv,
                file_name="filtered_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded file must contain a 'text' column.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
