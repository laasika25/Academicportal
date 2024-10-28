import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
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

# Preprocessing
try:
    # Ensure all data is string type
    df['Student Query'] = df['Student Query'].astype(str)
    df['Answer'] = df['Answer'].astype(str)
    
    # Vectorizing text data using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Student Query'])
    
    # Create a Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(tfidf_matrix)
    
    logging.info("Preprocessing and model creation completed successfully.")

except Exception as e:
    logging.error(f"Error during preprocessing: {str(e)}")
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
        
        # Vectorize the user query
        user_query_tfidf = vectorizer.transform([user_query])
        
        # Find the nearest neighbor
        distances, indices = nn_model.kneighbors(user_query_tfidf)
        
        # Get the answer for the nearest neighbor
        nearest_query_index = indices[0][0]
        answer = df.iloc[nearest_query_index]['Answer']
        
        logging.info(f"Found answer: {answer}")
        return jsonify({'response': answer})
    
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
