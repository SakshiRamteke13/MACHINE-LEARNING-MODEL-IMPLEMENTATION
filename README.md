# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS
*NAME* : SAKSHI KAILASH RAMTEKE
*INTERN ID* : CT4MTDF290
*DOMAIN* : PYTHON PROGRAMMING
*DURATION* : 16 WEEKS
*MENTOR* : NEELA SANTHOSH KUMAR

This project implements a simple spam detection classifier using scikit-learn (TF-IDF + Multinomial Naive Bayes).
It demonstrates data loading, preprocessing, training, evaluation, and saving model artifacts for later prediction.

## What's included
- `spam_data.csv` - sample labeled messages (ham/spam)
- `train_model.py` - trains the classifier and saves artifacts to `outputs/`
- `predict_cli.py` - CLI predictor that uses saved artifacts
- `streamlit_app.py` - optional Streamlit web UI for predictions
- `requirements.txt` - Python packages required
- `outputs/` - folder where model & metrics will be saved

## Step-by-step setup (from scratch)

### 1) Install Python 3.10+
- Download and install from https://www.python.org/downloads/
- On Windows check 'Add Python to PATH' during installation.

### 2) Recommended: Visual Studio Code
- https://code.visualstudio.com/

### 3) Create & activate a virtual environment
Open a Terminal in the project folder and run:

Windows PowerShell:
```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```
macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4) Install dependencies
```bash
pip install -r requirements.txt
```

### 5) Train the model
```bash
python train_model.py
```
This will create files under `outputs/`:
- `spam_model.joblib`
- `tfidf_vectorizer.joblib`
- `metrics.txt`
- `confusion_matrix.png`

### 6) Run predictions (CLI)
```bash
python predict_cli.py
```

### 7) Run Streamlit web UI (optional)
```bash
streamlit run streamlit_app.py
```

## How it works (brief)
1. Data is loaded from `spam_data.csv` with columns `label,message`.
2. Text is preprocessed (lowercase, remove punctuation).
3. TfidfVectorizer transforms text to numeric feature vectors.
4. Multinomial Naive Bayes is trained on the vectors.
5. Model and vectorizer saved with joblib for later inference.

## Extending & improving
- Use a larger real-world dataset (e.g., SMS Spam Collection) for better performance.
- Try different models: Logistic Regression, SVM, or ensemble methods.
- Add more preprocessing: spelling correction, stemming/lemmatization, handling URLs/emails.
- Use embeddings (sentence-transformers) for semantic understanding.

## Troubleshooting
- If you see `ModuleNotFoundError`, ensure venv activated and `pip install -r requirements.txt` succeeded.
- If Streamlit port conflicts, use `--server.port` to change the port.

## Output

[demo_run.txt](https://github.com/user-attachments/files/22476926/demo_run.txt)

