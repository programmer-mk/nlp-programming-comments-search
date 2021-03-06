from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix
import sys
sys.path.append("../data_preprocessing")
from .preprocessing import tf_idf, frequency_filtering, bigrams, trigrams

PROCESSED_DATA_DIR = '../../resources/classification-results'

operating_system = sys.platform
if operating_system == 'win32':
    PROCESSED_DATA_DIR = 'src\main/resources/classification-results'


def train_model(train, test, fold_no, processing_technique):

    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)

    score = f1_score(y_test, predictions, average='weighted')

    print('Fold',str(fold_no),'F1 SCORE:', score)
    f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-fold-{fold_no}-nb.txt", "a")
    f.write("\n")
    f.write(f"'Fold',{str(fold_no)},'F1 SCORE:',{score}")
    f.write("\n")
    f.write(f"{confusion_matrix(y_test, predictions, labels=['0','1','2','3'])}")
    f.close()

    return score, mnb


def train_naive_bayes(data_frame, processing_technique):
    skf = StratifiedKFold(n_splits=10)
    data_frame.dropna(how='any', inplace=True)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    average = 0
    

    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.iloc[train_index,:]
        test = data_frame.iloc[test_index,:]

        if processing_technique == 'TF-IDF':
            train_preprocessed, vectorizer = tf_idf(train.copy(), False)
            test_preprocessed, _ = tf_idf(test.copy(), False, vectorizer)
            train_preprocessed['SimilarityScore'] = train['SimilarityScore'].values
            test_preprocessed['SimilarityScore'] = test['SimilarityScore'].values

        elif processing_technique == 'frequency_filtering':

            train_preprocessed, vectorizer = frequency_filtering(train.copy(), False)
            test_preprocessed, _ = frequency_filtering(test.copy(), False, vectorizer)
            train_preprocessed['SimilarityScore'] = train['SimilarityScore'].values
            test_preprocessed['SimilarityScore'] = test['SimilarityScore'].values

        elif processing_technique == 'bigrams':

            train_preprocessed, vectorizer = bigrams(train.copy(), False)
            test_preprocessed, _ = bigrams(test.copy(), False, vectorizer)
            train_preprocessed['SimilarityScore'] = train['SimilarityScore'].values
            test_preprocessed['SimilarityScore'] = test['SimilarityScore'].values

        elif processing_technique == 'trigrams':

            train_preprocessed, vectorizer = trigrams(train.copy(), False)
            test_preprocessed, _ = trigrams(test.copy(), False, vectorizer)
            train_preprocessed['SimilarityScore'] = train['SimilarityScore'].values
            test_preprocessed['SimilarityScore'] = test['SimilarityScore'].values

        else:
            train_preprocessed = train
            test_preprocessed = test

        score, mnb = train_model(train_preprocessed,test_preprocessed,fold_no, processing_technique)
        average += score
        fold_no += 1

    print("Average F1 SCORE: of Naive Bayes is {:.2f}%".format(average / 10))
    f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-fold-average-nb.txt", "a")
    f.write("\n")
    f.write("Average F1 SCORE of Logistic Regression is {:.2f}%".format(average / 10))
    f.write("\n")
    f.close()

    return mnb


def naive_bayes_classifier(comments_data, processing_technique):
    print(f"Naive bayes classifier {processing_technique} data")
    mnb = train_naive_bayes(comments_data, processing_technique)
    return mnb
