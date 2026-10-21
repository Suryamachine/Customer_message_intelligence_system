from biagram_words import biagram_model
from uni_biagram_predict import new_messages

probabilities = biagram_model.predict_proba(new_messages)
classes = biagram_model.classes_

for message, scores in zip(new_messages, probabilities):
    print("\nMessage:", message)

    ranked = sorted(
        zip(classes, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    for category, score in ranked:
        print(category, "=", round(score, 3))

        