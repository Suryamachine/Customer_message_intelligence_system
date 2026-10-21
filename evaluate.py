from sklearn.metrics import accuracy_score
from biagram_words import unigram_model
from test_split import X_test,y_test
import pandas as pd


uni_biagram_predictions = unigram_model.predict(X_test)

unigram_accuracy = accuracy_score(y_test, uni_biagram_predictions)

print("\nUnigram accuracy:", round(unigram_accuracy, 3))

results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": uni_biagram_predictions,
})

print("\nTest predictions:")
print(results.to_string(index=False))



