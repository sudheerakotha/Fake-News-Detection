# fake_news_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv("fake_or_real_news.csv")  # Provide the correct path here

print("\nSample Data:")
print(df[['title', 'text', 'label']].head())

# 2. Data Preprocessing
print("\nPreprocessing data...")
df = df[['text', 'label']]
df.dropna(inplace=True)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# 3. Splitting data into training and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Vectorization using TF-IDF
print("\nVectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# 5. Training the Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_tf, y_train)

# 6. Making predictions
print("\nPredicting test data...")
y_pred = model.predict(X_test_tf)

# 7. Evaluation metrics
print("\nEvaluation Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 8. Plot confusion matrix using matplotlib only
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Fake", "Real"])
plt.yticks(tick_marks, ["Fake", "Real"])

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
