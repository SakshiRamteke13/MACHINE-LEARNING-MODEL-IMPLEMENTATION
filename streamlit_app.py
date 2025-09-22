# streamlit_app.py - Web UI for Spam Detector
import streamlit as st
import joblib, os, re

MODEL_PATH = 'outputs/spam_model.joblib'
VECT_PATH = 'outputs/tfidf_vectorizer.joblib'

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title='Spam Detector', layout='centered')
st.title('Task-4: Spam Detector (TF-IDF + Naive Bayes)')

st.write('This demo classifies messages as **spam** or **ham**. Train the model with `python train_model.py` first.')

if not os.path.exists(MODEL_PATH):
    st.warning('Model not found. Please run `python train_model.py` to train and save the model.')

user_input = st.text_area('Enter a message to classify', height=150)
if st.button('Predict'):
    if not os.path.exists(MODEL_PATH):
        st.error('Model not found. Train the model before predicting.')
    else:
        model = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
        proc = simple_preprocess(user_input)
        vec = vect.transform([proc])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        label = 'spam' if pred==1 else 'ham'
        confidence = prob[pred]
        st.success(f'Prediction: {label} (confidence={confidence:.3f})')
