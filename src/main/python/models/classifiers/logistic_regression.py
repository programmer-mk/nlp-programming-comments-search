from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
from .train_helper import train_test_init


def train_model(train, test, fold_no, rf, c=1):
    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    if rf == "l1":
        solver = 'liblinear'
        svc = LogisticRegression(penalty=rf, C=c, solver=solver, multi_class='ovr', max_iter=15000)
    else:
        solver = 'lbfgs'
        svc = LogisticRegression(penalty=rf, C=c, solver=solver, multi_class='ovr', max_iter=15000, n_jobs=-1)

    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    score = accuracy_score(y_test,predictions)
    print('Fold',str(fold_no),'Accuracy:', score)
    return score


def compare_regularisation_functions(data_frame, rf):
    skf = StratifiedKFold(n_splits=10)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    average = 0
    for train_index, test_index in skf.split(data_frame, target):
        train = data_frame.iloc[train_index,:]
        test = data_frame.iloc[test_index,:]
        score = train_model(train,test,fold_no, rf)
        average += score
        fold_no += 1
    print("Average score of Logistic Regression is {:.2f}%".format(average / 10))


def optimize_c_parameter(train, test):
    X_train, y_train, X_test, y_test = train_test_init(train, test)

    # defining parameter range
    # values above 10 like [100, 1000] takes a lot of time, won't be considered
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

    # Refit an estimator using the best found parameters on the whole dataset.
    grid = GridSearchCV(LogisticRegression(max_iter=25000, multi_class='ovr',  n_jobs=-1), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train.values.ravel())

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


def logistic_regression(comments_data):
    print("Logistic regression classifier")

    # Testing differences between regularisation functions
    print("> L1/L2 comparing")
    compare_regularisation_functions(comments_data, 'l1')
    compare_regularisation_functions(comments_data, 'l2')

    # Optimizing C parameter
    print("> Results with optimized C parameter")
    train, test = train_test_split(comments_data, test_size=0.2, random_state=42, shuffle=True)
    optimize_c_parameter(train, test)

