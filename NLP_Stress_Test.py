from biagram_words import biagram_model
tricky_messages = [
    "I can login but my payment keeps failing",
    "My order arrived but the device does not work",
    "Great, another package that never arrived",
    "payment deduct ayyindi but order confirm avvaledu",
    "I need help with something",
    "my app crashed and my payment was failed "
    
]

tricky_predictions = biagram_model.predict(tricky_messages)

for message, prediction in zip(tricky_messages, tricky_predictions):
    print("\nMessage:", message)
    print("Prediction:", prediction)

