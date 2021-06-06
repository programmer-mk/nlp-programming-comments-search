def train_test_init(train, test):
    y = ['SimilarityScore']
    y_train = train[y]
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)
    return X_train, y_train, X_test, y_test

    