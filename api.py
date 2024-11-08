import pickle
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Load the model and vectorizer
with open('model_nb.pkl', 'rb') as file_1:
    model_nb = pickle.load(file_1)

with open('vectorizor.pkl', 'rb') as file_2:
    vectorizer = pickle.load(file_2)

# Initialize stopwords and stemmer
stpwds_id = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

# Initialize FastAPI app
app = FastAPI()

# Define the request body model
class TextData(BaseModel):
    text: str

# Preprocessing function
def text_preprocessing(text: str) -> str:
    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub("@[A-Za-z0-9_]+", " ", text)

    # Hashtags removal
    text = re.sub("#[A-Za-z0-9_]+", " ", text)

    # Newline removal (\n)
    text = re.sub(r"\\n", " ", text)

    # Whitespace removal
    text = text.strip()

    # Repeat letters removal
    text = re.sub(r'(\w)\1+\b', r'\1', text)

    # URL removal
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)

    # Non-letter removal
    text = re.sub(r"[^A-Za-z\s']", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_id]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Combine tokens
    text = ' '.join(tokens)

    return text

# Prediction endpoint
@app.post("/predict")
async def predict(text_data: TextData):
    text = text_data.text

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Preprocess text
    preprocessed_text = text_preprocessing(text)

    df = pd.DataFrame({"random": [preprocessed_text]})

    # # Vectorize text
    vectorized_text = vectorizer.transform(df['random'])

    # Make prediction
    prediction = model_nb.predict_proba(vectorized_text)
    
    if prediction[0][0] > 0.25:
        hasil = 0
    else:
        hasil = 1

    return {
        "original_text": text,
        "preprocessed_text": preprocessed_text,
        "prediction": "Pajak" if hasil == 1 else 'Non-Pajak'
    }