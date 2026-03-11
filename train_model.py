import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = {
    "text": [
        "Government announces new education policy",
        "Aliens landed in Karachi last night",
        "Stock market increases today",
        "Miracle cure for cancer found",
        "Prime minister visits hospital",
        "Fake news about celebrity death spreads"
    ],
    "label": [1, 0, 1, 0, 1, 0]  
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")