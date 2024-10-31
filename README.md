# ML Model for Academic Purposes

A machine learning-powered academic chatbot designed to assist students by providing answers to frequently asked questions and facilitating academic inquiries. This chatbot leverages natural language processing (NLP) with Sentence-BERT for efficient question-answering.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Methodology](#model-and-methodology)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [License](#license)

## Introduction

The "ML Model for Academic Purposes" is designed to enhance student support by answering academic-related questions in real-time. The chatbot takes a query as input, computes similarity with pre-embedded queries from a dataset, and returns the most relevant answer.

## Features

- **Natural Language Processing:** Understands and processes academic questions using Sentence-BERT embeddings.
- **Cosine Similarity Matching:** Finds the best-matching response from a dataset based on cosine similarity.
- **User-Friendly Interface:** Responsive and easy-to-use web interface built with Flask and HTML/CSS.
- **Error Handling:** Provides informative error messages to users when issues arise.

## Installation

### Prerequisites
- Python 3.10 or higher
- Required libraries in `requirements.txt`

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ML-Model-for-Academic-Purposes.git
   cd ML-Model-for-Academic-Purposes
