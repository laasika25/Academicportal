import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from flask import Flask, request, render_template, jsonify
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
try:
    df = pd.read_excel('dataset.xlsx')
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"First few rows: \n{df.head()}")
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    sys.exit(1)

# Create the machine learning model
try:
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['Student Query'], df['Answer'])
    logging.info("Model trained successfully")
except Exception as e:
    logging.error(f"Error training model: {str(e)}")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form['query']
        logging.info(f"Received query: {user_query}")
        response = model.predict([user_query])[0]
        logging.info(f"Model response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)