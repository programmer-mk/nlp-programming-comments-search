from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from .train_helper import train_test_init
from imblearn.over_sampling import SMOTE
import pandas as pd
import sys
sys.path.append("../data_preprocessing")
from preprocessing import tf_idf

PROCESSED_DATA_DIR = '../../resources/classification-results'

def train_model(train, test, fold_no, rf, processing_technique, c=0.01):
    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', ['0','1','2','3'], y_train)

    class_weights = {
        '0': weights[0],
        '1': weights[1] * 12,
        '2': weights[2] * 12,
        '3':weights[3] * 12
    }


    print(f'weights are: {weights}')
    if rf == "l1":
        solver = 'liblinear'
        svc = LogisticRegression(penalty=rf, C=c, solver = solver, max_iter=200,verbose=10, random_state=42 , multi_class='ovr', n_jobs=-1, class_weight=class_weights)
    else:
        solver = 'lbfgs'
        svc = LogisticRegression(penalty='l2', random_state=42, C=c, n_jobs=-1, solver=solver, multi_class='ovr',
                                 max_iter=80,verbose=10, class_weight='balanced')

    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    print(confusion_matrix(y_test, predictions, labels=['0','1','2','3']))
    score = f1_score(y_test, predictions, average='weighted')

    f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-{rf}-fold-{fold_no}-lg.txt", "a")
    f.write("\n")
    f.write(f"'Fold',{str(fold_no)},'F1 SCORE:',{score}")
    f.write("\n")
    f.write(f"{confusion_matrix(y_test, predictions, labels=['0','1','2','3'])}")
    f.close()

    print('Fold',str(fold_no),'F1 SCORE:', score)
    return score


def compare_regularisation_functions(data_frame, rf, processing_technique):
    data_frame.dropna(how='any', inplace=True)
    skf = StratifiedKFold(n_splits=10)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    average = 0
    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.iloc[train_index,:]
        test = data_frame.iloc[test_index,:]

        if processing_technique == 'TF-IDF':
            train_preprocessed, vectorzer = tf_idf(train.copy(), False)
            test_preprocessed, _ = tf_idf(test.copy(), False, vectorzer)

            train_preprocessed['SimilarityScore'] = train['SimilarityScore'].values
            test_preprocessed['SimilarityScore'] = test['SimilarityScore'].values
        else:
            train_preprocessed = train
            test_preprocessed = test

        score = train_model(train_preprocessed,test_preprocessed,fold_no, rf, processing_technique)
        average += score
        fold_no += 1
    f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-{rf}-fold-average-lg.txt", "a")
    f.write("\n")
    f.write("Average F1 SCORE of Logistic Regression is {:.2f}%".format(average / 10))
    f.write("\n")
    f.close()
    print("Average F1 SCORE of Logistic Regression is {:.2f}%".format(average / 10))


def optimize_c_parameter(train, test, processing_technique):
    X_train, y_train, X_test, y_test = train_test_init(train, test)

    if processing_technique == 'TF-IDF':
        X_train_preprocessed, vectorzer = tf_idf(X_train.copy(), False)
        X_test_preprocessed, _ = tf_idf(X_test.copy(), False, vectorzer)
    else:
        X_train_preprocessed = X_train
        X_test_preprocessed = X_test

    # defining parameter range
    # values above 10 like [100, 1000] takes a lot of time, won't be considered
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1]}

    # Refit an estimator using the best found parameters on the whole dataset.
    grid = GridSearchCV(LogisticRegression(penalty='l1', max_iter=25000, multi_class='ovr', solver = 'liblinear', class_weight='balanced'), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train_preprocessed, y_train.values.ravel())

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test_preprocessed)

    # print classification report
    print(classification_report(y_test, grid_predictions))


def logistic_regression_classifier(comments_data, processing_technique):
    print(f"Logistic regression classifier, {processing_technique} data")

    # Testing differences between regularisation functions
    print("> L1/L2 comparing")
    compare_regularisation_functions(comments_data, 'l1', processing_technique)
    compare_regularisation_functions(comments_data, 'l2', processing_technique)

    # Optimizing C parameter
    print(f"> Results with optimized C parameter, {processing_technique} data")
    train, test = train_test_split(comments_data, test_size=0.2, random_state=42, shuffle=True)
    optimize_c_parameter(train, test, processing_technique)
