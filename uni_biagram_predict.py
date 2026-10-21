
from unigram_words import unigram_model
from biagram_words import biagram_model

new_messages = [
    "Money disappeared from my card after checkout",
    "I cannot get into my account",
    "My parcel should have arrived yesterday",
    "The app closes as soon as I open it",
    "I need my money back",
    "The sign in code is missing",
]

unigram_new = unigram_model.predict(new_messages)
bigram_new = biagram_model.predict(new_messages)

for message, uni, bi in zip(new_messages, unigram_new, bigram_new):
    print("\nMessage:", message)
    print("Unigram prediction:", uni)
    print("Unigram + Bigram prediction:", bi)

