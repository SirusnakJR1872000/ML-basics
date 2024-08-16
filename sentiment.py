#  Basic Sentiment Analysis Model Using TF-IDF and SVM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Example dataset (sentences and their corresponding labels)
sentences = [
    "I love this product!",
    "This is the worst experience ever.",
    "I am very happy with this service.",
    "I hate this place.",
    "This is amazing!",
    "I am disappointed."
]

# Labels: 1 for positive, 0 for negative
labels = [1, 0, 1, 0, 1, 0]

# Step 1: Convert sentences to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 3: Train an SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Predicted labels:", y_pred)
print("Accuracy:", accuracy)
