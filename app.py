from flask import Flask, render_template, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Load the CountVectorizer and TfidfTransformer from training
bow_transformer = joblib.load('bow_transformer.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')


@app.route('/')
def index():
    return render_template('index.html',prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the text (apply the same preprocessing as in your training script)
        text = text.replace('<br />', '')  # Remove HTML tags
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation

        # Vectorize the preprocessed text
        text_bow = bow_transformer.transform([text])

        # Transform the vectorized text using the TF-IDF transformer
        text_tfidf = tfidf_transformer.transform(text_bow)

        # Make predictions using the model
        prediction = model.predict(text_tfidf)

        # Determine sentiment label
        sentiment = "Positive" if prediction[0] == 'positive' else "Negative"

        # Return the result as JSON
        return jsonify({'sentiment': sentiment})


if __name__ == '__main__':
    app.run(debug=True)
