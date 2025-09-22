# train_model.py - Train a spam classifier using TF-IDF + MultinomialNB
import pandas as pd
import re, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

DATA_CSV = 'spam_data.csv'
OUT_DIR = 'outputs'

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    df['text'] = df['message'].apply(simple_preprocess)
    X = df['text'].tolist()
    y = df['label'].map({'ham':0, 'spam':1}).values

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # vectorize
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    # model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham','spam'])
    cm = confusion_matrix(y_test, y_pred)

    print('Accuracy:', acc)
    print('Classification report:\n', report)
    print('Confusion matrix:\n', cm)

    # save artifacts
    joblib.dump(model, os.path.join(OUT_DIR, 'spam_model.joblib'))
    joblib.dump(vect, os.path.join(OUT_DIR, 'tfidf_vectorizer.joblib'))

    # save metrics to file
    with open(os.path.join(OUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(f'Accuracy: {acc}\n\n')
        f.write(report)
        f.write('\nConfusion matrix:\n')
        f.write(str(cm))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['ham','spam']); ax.set_yticklabels(['ham','spam'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    import numpy as np
    main()
