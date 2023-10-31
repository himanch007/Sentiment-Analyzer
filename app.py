from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = FastAPI()


@app.get("/nb-lr-comparision-for-amazon-reviews")
def navie_bayes_and_logistic_regression_model_comparision():

    # Load the data from the CSV file (amazon_reviews.csv)
    df = pd.read_csv("amazon_reviews.csv")

    # Handle missing values in the "reviewText" column by filling them with an empty string
    df['reviewText'] = df['reviewText'].fillna('')

    # Assuming "overall" column contains the sentiment label (0 or 1)
    X = df['reviewText']
    y = df['overall']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TF-IDF vectors from the text data
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Initialize and train the Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Predict sentiment using the Naive Bayes classifier
    nb_predictions = nb_classifier.predict(X_test_tfidf)

    # Initialize and train the Logistic Regression classifier
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    lr_classifier.fit(X_train_tfidf, y_train)

    # Predict sentiment using the Logistic Regression classifier
    lr_predictions = lr_classifier.predict(X_test_tfidf)

    # Calculate metrics for Naive Bayes
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    nb_precision = precision_score(y_test, nb_predictions, average='weighted', zero_division=0)
    nb_recall = recall_score(y_test, nb_predictions, average='weighted', zero_division=0)
    nb_f1 = f1_score(y_test, nb_predictions, average='weighted', zero_division=0)

    # Calculate metrics for Logistic Regression
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_precision = precision_score(y_test, lr_predictions, average='weighted', zero_division=0)
    lr_recall = recall_score(y_test, lr_predictions, average='weighted', zero_division=0)
    lr_f1 = f1_score(y_test, lr_predictions, average='weighted', zero_division=0)


    return {
        "Naive Bayes Metrics": {
            "Accuracy": nb_accuracy,
            "Precision": nb_precision,
            "Recall": nb_recall,
            "F1-score": nb_f1
        },
        "Logistic Regression Metrics": {
            "Accuracy": lr_accuracy,
            "Precision": lr_precision,
            "Recall": lr_recall,
            "F1-score": lr_f1
        }
    }


@app.get("/get-comment-sentiment")
def get_comment_sentiment(comment: str):

    # Download the VADER lexicon if you haven't already
    nltk.download('vader_lexicon')

    # Initialize the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get the sentiment scores for the comment
    sentiment_scores = analyzer.polarity_scores(comment)

    # Determine the sentiment based on the compound score
    compound_score = sentiment_scores['compound']

    message = ''
    if compound_score >= 0.05:
        message = "Positive"
    elif compound_score <= -0.05:
        message = "Negative"
    else:
        message = "Neutral"

    return {"message": message}


# @app.get("/nb-lr-comparision-for-imdb-reviews")
# def navie_bayes_and_logistic_regression_model_comparision():

    # Load the data from the CSV file
    # df = pd.read_csv("imdb_sup.csv")

    # # Split the data into training and testing sets
    # X = df['Review']
    # y = df['Sentiment']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create TF-IDF vectors from the text data
    # tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    # X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # # Save the TF-IDF vectorizer to a file for future use
    # joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

    # # Initialize and train the Naive Bayes classifier
    # nb_classifier = MultinomialNB()
    # nb_classifier.fit(X_train_tfidf, y_train)

    # # Predict sentiment using the Naive Bayes classifier
    # nb_predictions = nb_classifier.predict(X_test_tfidf)

    # # Initialize and train the Logistic Regression classifier
    # lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    # lr_classifier.fit(X_train_tfidf, y_train)

    # # Predict sentiment using the Logistic Regression classifier
    # lr_predictions = lr_classifier.predict(X_test_tfidf)

    # # Save the lr_classifier to a file for future use
    # joblib.dump(lr_classifier, "lr_classifier.pkl")

    # # Calculate metrics for Naive Bayes
    # nb_accuracy = accuracy_score(y_test, nb_predictions)
    # nb_precision = precision_score(y_test, nb_predictions)
    # nb_recall = recall_score(y_test, nb_predictions)
    # nb_f1 = f1_score(y_test, nb_predictions)

    # # Calculate metrics for Logistic Regression
    # lr_accuracy = accuracy_score(y_test, lr_predictions)
    # lr_precision = precision_score(y_test, lr_predictions)
    # lr_recall = recall_score(y_test, lr_predictions)
    # lr_f1 = f1_score(y_test, lr_predictions)

    # return {"message": "Done"}


    # Comparing using trained model
    # try:
    #     # Load the TF-IDF vectorizer from the saved file
    #     tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    #     # Load the lr_classifier from the saved file
    #     lr_classifier = joblib.load("lr_classifier.pkl")
    # except:
    #     return {"message": "Please train the model first"}

    # # Function to predict sentiment for a comment
    # comment_tfidf = tfidf_vectorizer.transform([comment])
    # lr_prediction = lr_classifier.predict(comment_tfidf)[0]

    # # Map predictions to sentiment labels
    # sentiment_mapping = {0: "Negative", 1: "Positive"}

    # lr_sentiment = sentiment_mapping[lr_prediction]