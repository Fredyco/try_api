import pickle
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
from fastapi.responses import JSONResponse

# Download necessary NLTK data once
nltk.download('stopwords')
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI()

# Define the request body model
class TextData(BaseModel):
    text: str

# Preload model and vectorizer once
with open('model_nb.pkl', 'rb') as file_1:
    model_nb = pickle.load(file_1)

with open('vectorizor.pkl', 'rb') as file_2:
    vectorizer = pickle.load(file_2)

# Initialize stopwords and stemmer
stpwds_id = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

# Preprocessing function
def text_preprocessing(text: str) -> str:
    # Case folding (lowercase)
    text = text.lower()

    # Remove unwanted patterns (mentions, hashtags, newlines, urls, non-letters, etc.)
    text = re.sub(r"(@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|http\S+|www.\S+|[^A-Za-z\s'])", " ", text)

    # Tokenization and removal of stopwords and stemming
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stpwds_id]

    # Combine tokens into a string
    return ' '.join(tokens)

# Prediction endpoint
@app.post("/predict")
async def predict(text_data: TextData):
    text = text_data.text

    # Check if text is provided
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Preprocess the text
    preprocessed_text = text_preprocessing(text)

    # Create a DataFrame for vectorization
    df = pd.DataFrame({"random": [preprocessed_text]})

    # Vectorize the text
    vectorized_text = vectorizer.transform(df['random'])

    # Make prediction
    prediction = model_nb.predict_proba(vectorized_text)

    # Define threshold for prediction (e.g., if > 0.25, classify as "Pajak")
    if prediction[0][0] > 0.25:
        hasil = 0  # Non-Pajak
    else:
        hasil = 1  # Pajak

    return JSONResponse(content={
        "original_text": text,
        "preprocessed_text": preprocessed_text,
        "prediction": "Pajak" if hasil == 1 else 'Non-Pajak'
    })

