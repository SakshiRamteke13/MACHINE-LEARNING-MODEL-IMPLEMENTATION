# predict_cli.py - simple command-line predictor using saved model
import joblib, re, os
MODEL_PATH = 'outputs/spam_model.joblib'
VECT_PATH = 'outputs/tfidf_vectorizer.joblib'

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        print('Model or vectorizer not found. Please run train_model.py first to train and save artifacts.')
        return
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    print('Spam detector ready. Type a message (type "exit" to quit).')
    while True:
        text = input('\nMessage: ').strip()
        if text.lower() in ('exit','quit'):
            print('Goodbye!')
            break
        proc = simple_preprocess(text)
        vec = vect.transform([proc])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        label = 'spam' if pred==1 else 'ham'
        confidence = prob[pred]
        print(f'Prediction: {label} (confidence={confidence:.3f})')

if __name__ == '__main__':
    main()
