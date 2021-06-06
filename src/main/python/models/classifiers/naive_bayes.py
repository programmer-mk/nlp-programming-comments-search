from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from .train_helper import train_test_init

def train_model(train, test, fold_no):
    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    print('Fold', str(fold_no), 'Accuracy:', accuracy_score(y_test,predictions))


def train_naive_bayes(data_frame):
    skf = StratifiedKFold(n_splits=10)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.loc[train_index,:]
        test = data_frame.loc[test_index,:]
        train_model(train,test,fold_no)
        fold_no += 1

def naive_bayes_classifier(comments_data):
    print("Naive bayes classifier")
    train_naive_bayes(comments_data)
