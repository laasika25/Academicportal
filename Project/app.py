import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, render_template, jsonify
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
try:
    df = pd.read_excel('dataset.xlsx')
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")

    # Check for required columns
    if 'Student Query' not in df.columns or 'Answer' not in df.columns:
        logging.error("Missing 'Student Query' or 'Answer' columns in dataset.")
        raise ValueError("Dataset must contain 'Student Query' and 'Answer' columns.")

    # Check for missing data
    if df['Student Query'].isnull().any() or df['Answer'].isnull().any():
        logging.error("Missing values in 'Student Query' or 'Answer' columns.")
        raise ValueError("There are missing values in the dataset.")

except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# Load the Sentence-BERT model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence-BERT model loaded successfully.")

    # Encode all queries in the dataset
    df['Query Embedding'] = df['Student Query'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    logging.info("Embeddings for all queries created successfully.")

except Exception as e:
    logging.error(f"Error during model loading or embedding: {str(e)}")
    raise

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

        if not user_query:
            logging.error("Empty query received.")
            return jsonify({'error': 'Empty query received'}), 400

        # Encode the user query
        user_query_embedding = model.encode(user_query, convert_to_tensor=True)

        # Compute cosine similarity with all query embeddings in the dataset
        similarities = [util.cos_sim(user_query_embedding, query_emb)[0][0].item() for query_emb in df['Query Embedding']]
        
        # Find the index of the highest similarity score
        nearest_query_index = similarities.index(max(similarities))
        answer = df.iloc[nearest_query_index]['Answer']

        logging.info(f"Found answer: {answer}")
        return jsonify({'response': answer})

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
