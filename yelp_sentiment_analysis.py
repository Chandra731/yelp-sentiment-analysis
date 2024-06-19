import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load Yelp dataset (replace 'your_yelp_data.csv' with the actual path)
yelp = pd.read_csv('your_yelp_data.csv')

# Filter for 1-star and 5-star reviews
yelp_class = yelp[yelp['stars'].isin([1, 5])]

# Split into features (X) and labels (y)
X = yelp_class['text']
y = yelp_class['stars']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# --- Traditional Process ---

# 1. Count Vectorization
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# 2. TF-IDF Transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# 3. Train Naive Bayes Model
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

# --- Pipeline Process ---

# Create a pipeline object
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# --- Prediction and Evaluation (both methods) ---

# Transform test data using the traditional process
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Predictions using both methods
predictions_traditional = nb_clf.predict(X_test_tfidf)
predictions_pipeline = pipeline.predict(X_test)

# Evaluation
print("\n--- Traditional Method Results ---")
print(confusion_matrix(y_test, predictions_traditional))
print(classification_report(y_test, predictions_traditional))

print("\n--- Pipeline Method Results ---")
print(confusion_matrix(y_test, predictions_pipeline))
print(classification_report(y_test, predictions_pipeline))
