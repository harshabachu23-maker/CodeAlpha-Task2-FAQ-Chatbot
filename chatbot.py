import json
import re

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download required NLTK data
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# Load FAQs
with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)


# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Preprocess FAQ questions
questions = [
    preprocess(faq["question"])
    for faq in faqs
]


# Convert questions to TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)


# Find the best matching answer
def get_answer(user_question):
    processed_question = preprocess(user_question)

    if not processed_question:
        return "Please enter a meaningful question."

    user_vector = vectorizer.transform([processed_question])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.20:
        return (
            "Sorry, I don't understand that question. "
            "Please ask something related to Artificial Intelligence."
        )

    return faqs[best_match_index]["answer"]


# Start chatbot
print("=" * 60)
print("             AI FAQ CHATBOT")
print("=" * 60)
print("Ask questions about Artificial Intelligence.")
print("Type 'exit' to close the chatbot.")
print("=" * 60)

while True:
    user_input = input("\nYou: ")

    if user_input.lower().strip() == "exit":
        print("Bot: Thank you for using the AI FAQ Chatbot!")
        break

    if not user_input.strip():
        print("Bot: Please enter a question.")
        continue

    answer = get_answer(user_input)

    print("Bot:", answer)
