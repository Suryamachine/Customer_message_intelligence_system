
from unigram_words import unigram_model
from biagram_words import biagram_model

new_messages = [
    "I need a refund for this purchase",
    "The invoice amount is incorrect",
    "My card payment failed",
    "Why did you charge me extra",
    "I was billed for something I did not buy",
    "The refund has not reached my account",
    "My transaction was declined",
    "The payment page charged me twice",
    "I forgot my password",
    "I cannot sign in to my account",
    "My login is not working",
    "The verification code never arrived",
    "My account is locked",
]

unigram_new = unigram_model.predict(new_messages)
bigram_new = biagram_model.predict(new_messages)

for message, uni, bi in zip(new_messages, unigram_new, bigram_new):
    print("\nMessage:", message)
    print("Unigram prediction:", uni)
    print("Unigram + Bigram prediction:", bi)

