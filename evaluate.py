from sklearn.metrics import accuracy_score
from unigram_words import unigram_model
from test_split import X_test,y_test
import pandas as pd


unigram_predictions = unigram_model.predict(X_test)

unigram_accuracy = accuracy_score(y_test, unigram_predictions)

print("\nUnigram accuracy:", round(unigram_accuracy, 3))

results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": unigram_predictions,
})

print("\nTest predictions:")
print(results.to_string(index=False))

