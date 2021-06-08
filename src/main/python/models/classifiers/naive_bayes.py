from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score


def train_model(train, test, fold_no):
    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)

    #score = f1_score(y_test, predictions, average='weighted')
    score = accuracy_score(y_test,predictions)
    print('Fold',str(fold_no),'Accuracy:', accuracy_score(y_test,predictions))
    #print(f'Fold {fold_no}, f1_score: {score}')
    return score


def train_naive_bayes(data_frame):
    skf = StratifiedKFold(n_splits=10)
    data_frame.dropna(how='any', inplace=True)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    average = 0
    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.iloc[train_index,:]
        test = data_frame.iloc[test_index,:]
        score = train_model(train,test,fold_no)
        average += score
        fold_no += 1
    print("Average score of Naive Bayes is {:.2f}%".format(average / 10))


def naive_bayes_classifier(comments_data, processing_technique):
    print(f"Naive bayes classifier {processing_technique} data")
    train_naive_bayes(comments_data)
