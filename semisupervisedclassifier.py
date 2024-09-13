from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

labeled_data = [
    ("This is a document about sports", "Sports"),
    ("This is a news article", "News"),
    ("Another document about sports", "Sports"),
    ("A text sample about politics", "Politics"),
    ("A document discussing music", "Music")
]
unlabeled_data = [
    "This document discusses machine learning",
    "Another document about music",
    "A short text sample"
]

all_data = [text for text, _ in labeled_data] + unlabeled_data
texts, labels = zip(*labeled_data)
vectorizer = TfidfVectorizer(max_features=500)
features = vectorizer.fit_transform(all_data)
features_dense = features.toarray()
all_labels = sorted(set(labels))
label_distributions = np.zeros((len(texts), len(all_labels)))
for i, label in enumerate(labels):
    label_distributions[i, all_labels.index(label)] = 1

X_train = features_dense[:len(texts)]
y_train = labels
X_test = X_train
y_test = y_train

y_train_indices = np.array([all_labels.index(label) for label in y_train])
semi_clf = LabelPropagation()
semi_clf.fit(X_train, y_train_indices)
predictions = semi_clf.predict(X_test)

accuracy = accuracy_score(np.array([all_labels.index(label) for label in y_test]), predictions)
precision = precision_score(np.array([all_labels.index(label) for label in y_test]), predictions, average='weighted')
recall = recall_score(np.array([all_labels.index(label) for label in y_test]), predictions, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#output
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
