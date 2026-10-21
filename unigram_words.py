from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from test_split import X_train,y_train


unigram_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

convert = unigram_model.fit(X_train, y_train)
print(y_train)



