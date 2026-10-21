from sklearn.model_selection import train_test_split

from read_clean import data


X = data["message"]
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.50,
    random_state=42,
    stratify=y,
)

# print('training : ',len(X_train))

# print('testing :',len(X_test))

print(len(X_test))
