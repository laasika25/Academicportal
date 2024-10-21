from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback  # For detailed error reporting

app = Flask(__name__)

# Load dataset from Excel file
data = pd.read_excel('dataset.xlsx')

# Ensure the columns are named correctly
data.columns = ['Query_Id', 'Student_Query', 'Intent_Label', 'Answer']  # Correct capitalization here

# Remove leading/trailing spaces in student queries
data['Student_Query'] = data['Student_Query'].str.strip()

# Vectorize the student queries
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Student_Query'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")  # Check the shape of the TF-IDF matrix

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.json['question'].strip()  # Strip spaces from user input
        print(f"User Input: {user_input}")  # Print user input for debugging

        # Print dataset for verification
        print(f"Dataset: {data.head()}")  # Print first few rows of the dataset

        # Vectorize the user input
        user_input_vectorized = vectorizer.transform([user_input])

        # Compute similarities
        similarities = cosine_similarity(user_input_vectorized, tfidf_matrix)

        # Print similarities for debugging
        print(f"Similarities: {similarities}")

        # Find the index of the most similar query
        index = similarities.argsort()[0][-1]

        # Print the index and answer
        print(f"Index: {index}")
        answer = data['Answer'].iloc[index]
        print(f"Bot Answer: {answer}")

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error: {e}")  # Print the error for debugging
        return jsonify({'answer': 'Sorry, something went wrong.'})


if __name__ == '__main__':
    app.run(debug=True)
