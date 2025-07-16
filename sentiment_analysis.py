import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.utils.multiclass import unique_labels


# Ensure required resources are available
nltk.download('stopwords')

# Sample data
data = {
    'review': [
        "I love this product!",
        "Terrible customer service.",
        "Okay product, nothing special.",
        "Excellent quality!",
        "Worst experience ever.",
        "Absolutely fantastic and smooth!",
        "Do not recommend this at all.",
        "Good value for the price.",
        "I'm disappointed.",
        "Top-notch performance!"
    ]
}
df = pd.DataFrame(data)

# Load stopwords once
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase, remove non-letters, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

def get_sentiment(text):
    """Assign sentiment using TextBlob polarity."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Clean and label data
df['cleaned_review'] = df['review'].apply(clean_text)
df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# ✅ Train-test split (NO stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.utils.multiclass import unique_labels

labels = unique_labels(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred, labels=labels, target_names=le.inverse_transform(labels)))

# Save model artifacts
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
