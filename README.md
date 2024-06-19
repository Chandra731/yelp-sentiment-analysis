# Yelp Sentiment Analysis

This project explores sentiment analysis of Yelp reviews using machine learning. The goal is to classify reviews as positive or negative based on their text content.

## Dataset

The project uses the Yelp reviews dataset, which includes:

* Star rating (1-5)
* Review text
* Other user-related information (optional for exploration)

## Methodology

1. **Data Exploration:** Analyze review length distributions and other features.
2. **Feature Engineering:** Extract text length and potentially other relevant features.
3. **Model Training:** Train a Naive Bayes classifier on a subset of reviews (e.g., only 1-star and 5-star reviews).
4. **Model Evaluation:** Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
5. **Pipeline Implementation:** Explore using pipelines to streamline the workflow (and potentially troubleshoot!).

## Key Findings

* **5-star reviews tend to be longer than 1-star reviews.**
* **The "coolness" of a reviewer does not significantly correlate with the sentiment of their review.**
* **Naive Bayes can be a good starting point, but may require further refinement for better accuracy.**
* **Pipelines can be helpful but might require careful consideration for optimal results.**

## How to Run

1. **Clone this repository:**  `git clone https://github.com/brucelee31072004/yelp-sentiment-analysis.git`
2. **Install dependencies:** `pip install pandas scikit-learn matplotlib seaborn`
3. **Run the script:** `python yelp_sentiment_analysis.py`

## Future Work

* **Advanced NLP Techniques:** Experiment with TF-IDF, word embeddings, or other methods to improve model performance.
* **More Complex Models:**  Try different classifiers (e.g: SVM) or even deep learning approaches.
* **Domain-Specific Analysis:** Explore sentiment analysis within specific industries or categories of Yelp reviews.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
