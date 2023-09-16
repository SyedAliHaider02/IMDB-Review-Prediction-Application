import pandas as pd
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Define a function to remove HTML line breaks
def remove_br_tags(text):
    cleaned_text = text.replace('<br />', '')
    return cleaned_text

# Define a function to remove punctuation
def nopunct(text):
    nopunc = [x for x in text if x not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc

# Define a function to convert slang to synonyms
nltk.download('punkt')
nltk.download('wordnet')
def convert_slang(text):
    words = word_tokenize(text)
    converted_words = []

    for word in words:
        # Get synonyms for the current word
        synonyms = wordnet.synsets(word)
        
        # Use the first synonym as replacement (if available)
        replacement = synonyms[0].lemmas()[0].name() if synonyms else word
        converted_words.append(replacement)

    converted_text = " ".join(converted_words)
    return converted_text

# Apply data preprocessing steps
df['review'] = df['review'].apply(remove_br_tags)
df['review'] = df['review'].apply(nopunct)
df['review'] = df['review'].apply(convert_slang)

# Vectorize the text data using CountVectorizer
bow_transformer = CountVectorizer(analyzer='word').fit(df['review'])
reviews_bow = bow_transformer.transform(df['review'])

# Transform the bag-of-words into TF-IDF features
tfidf_transformer = TfidfTransformer().fit(reviews_bow)
reviews_tfidf = tfidf_transformer.transform(reviews_bow)

# Save the CountVectorizer (bow_transformer) to a .pkl file
bow_transformer_filename = 'bow_transformer.pkl'
joblib.dump(bow_transformer, bow_transformer_filename)

# Save the TfidfTransformer (tfidf_transformer) to a .pkl file
tfidf_transformer_filename = 'tfidf_transformer.pkl'
joblib.dump(tfidf_transformer, tfidf_transformer_filename)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB().fit(reviews_tfidf, df['sentiment'])

# Save the trained model to a .pkl file
model_filename = 'sentiment_model.pkl'
joblib.dump(model, model_filename)


