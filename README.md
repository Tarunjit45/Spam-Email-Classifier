# Spam Email Classifier

This project is a Spam Message Classifier built using Python and Streamlit. It uses a machine learning model to classify messages as spam or not spam.

## Requirements

- Python 3.11 or higher
- The following Python packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - streamlit

You can install the required packages using pip:

```
py -m pip install pandas numpy scikit-learn joblib streamlit
```

## Running the App

To run the Streamlit app, execute the following command in the project directory:

```
py -m streamlit run app.py
```

This will start the app locally. You can access it in your browser at:

```
http://localhost:8501
```

## Model Files

Make sure the following model files are present in the project directory:

- `spam_model.pkl`
- `vectorizer.pkl`

These files are required for the app to function correctly.

## Usage

Enter a message in the text area and click the "Predict" button to check if the message is spam or not.
