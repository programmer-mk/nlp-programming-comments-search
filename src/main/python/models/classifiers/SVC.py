import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, accuracy_score


def train_model(train, test, fold_no, rf, c=1):
    X = ['Retail_Price','Discount']
    y = ['Returned_Units']
    X_train = train[X]
    y_train = train[y]
    X_test = test[X]
    y_test = test[y]

    svc = LinearSVC(penalty=rf, C=c, dual=rf == 'l2', max_iter=15000)
    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    print('Fold',str(fold_no),'Accuracy:', accuracy_score(y_test,predictions))


def compare_regularisation_functions(data_frame, rf, c=1):
    skf = StratifiedKFold(n_splits=10)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.loc[train_index,:]
        test = data_frame.loc[test_index,:]
        train_model(train,test,fold_no, rf)
        fold_no += 1


def optimize_c_parameter(X_train, y_train, X_test, y_test):
    # defining parameter range
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    # Refit an estimator using the best found parameters on the whole dataset.
    grid = GridSearchCV(LinearSVC(), param_grid, refit=True, verbose=3, n_jobs=-1)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


def support_vector_classifier(comments_data, comments_similarity, type):
    """
    :param comments_data:  X_train + X_test
    :param comments_relevancy:  y_train + y_test
    :param type:
    :return:
    """
    print("Support vector classifier")
    data_frame = pd.read_csv(f'processed_data/{type}.txt')

    # Testing differences between regularisation functions
    print("> L1/L2 comparing")
    compare_regularisation_functions(data_frame, 'l1')
    compare_regularisation_functions(data_frame, 'l2')

    # Optimizing C parameter
    print("> Results with optimized C parameter")
    optimize_c_parameter()

