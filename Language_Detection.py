import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import tkinter as tk
from tkinter import Label, Text, Button, StringVar

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data from CSV file
file_path = "Language_Detection.csv"
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
df['Text'] = df['Text'].str.lower()
df['Text'] = df['Text'].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))
df['Text'] = df['Text'].apply(lambda text: re.sub(r'\d+', '', text))
df['Text'] = df['Text'].apply(lambda text: word_tokenize(text))
df['Text'] = df['Text'].apply(lambda tokens: [word for word in tokens if word not in stopwords.words('english')])
lemmatizer = WordNetLemmatizer()
df['Text'] = df['Text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
df['Text'] = df['Text'].apply(lambda tokens: ' '.join(tokens))

# Step 3: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['Text'])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, df['Language'], test_size=0.2, random_state=42)

# Step 5: Model Training
model = MultinomialNB()
model.fit(X_train, y_train)


def predict_language(text):
    # Preprocess input text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)

    # Transform using the same TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vectorized)
    return prediction[0]


class LanguageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Detection GUI")

        self.label = Label(root, text="Enter text:")
        self.label.pack()

        self.text_entry = Text(root, height=5, width=40)
        self.text_entry.pack()

        self.result_var = StringVar()
        self.result_label = Label(root, textvariable=self.result_var)
        self.result_label.pack()

        self.detect_button = Button(root, text="Detect Language", command=self.detect_language)
        self.detect_button.pack()

    def detect_language(self):
        input_text = self.text_entry.get("1.0", "end-1c")
        result = predict_language(input_text)
        self.result_var.set(f"Predicted Language: {result}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageDetectionApp(root)
    root.mainloop()