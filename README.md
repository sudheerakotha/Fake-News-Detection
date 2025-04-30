# Fake News Detection

This project aims to classify news articles as **fake** or **real** using machine learning techniques. We use **Logistic Regression** and **TF-IDF (Term Frequency-Inverse Document Frequency)** to preprocess the data and train a model to predict the authenticity of the news articles. The dataset used contains a collection of labeled news articles.

## Dataset

The dataset used in this project is the **Fake or Real News** dataset, which contains news articles labeled as either `FAKE` or `REAL`. It includes the following columns:

- `title`: The title of the news article.
- `text`: The content of the article.
- `label`: The label indicating whether the article is fake or real (FAKE = 0, REAL = 1).

## Requirements

The following Python libraries are required to run this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

To install them, use the following command:

```bash
pip install -r requirements.txt
