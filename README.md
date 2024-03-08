# Language Identification Model Project

## Overview

Welcome to the Language Identification Model project! This repository showcases an innovative language detection solution implemented in Python using a Multinomial Naive Bayes classifier and TF-IDF vectorization.

## Project Structure

- **`/Language Detection.ipynb`**: Jupyter notebook containing the project's code with detailed explanations and visualizations.
- **`Language_Detection.py`**: Python script with a graphical user interface (GUI) for easy text input and language detection.
- **`Language_Detection.csv`**: Dataset used for training and testing the language identification model.

## Getting Started

### Prerequisites

Make sure you have Python installed. Install the required packages using:

```bash
pip install scikit-learn pandas nltk
```

### Running the Jupyter Notebook

Explore the detailed implementation and analysis in `Language Detection.ipynb`

```bash
jupyter notebook Language Detection.ipynb
```

### Running the GUI Application

Execute the GUI script to experience real-time language detection.

```bash
python Language_Detection.py
```

## Project Workflow

1. **Data Preprocessing**: Text cleaning, tokenization, and lemmatization for improved model performance.
2. **Feature Extraction**: Utilizing TF-IDF vectorization to transform text data into numerical features.
3. **Model Training**: Employing a Multinomial Naive Bayes classifier for language prediction.
4. **GUI Application**: Building an interactive GUI for user-friendly language detection.

## Results

The model achieves a test accuracy of 65%, demonstrating its effectiveness in identifying languages. The GUI allows users to input text and receive predicted language labels.
The moderate test accuracy of 65% is due to the absence of enough training data

## Contribution

Feel free to contribute to enhance the model's accuracy, add features to the GUI, or address any issues you identify.

Thank you for exploring the Language Identification Model project! 🚀✨
