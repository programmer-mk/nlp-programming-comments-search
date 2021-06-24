from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, accuracy_score
from .train_helper import train_test_init


def train_model(train, test, fold_no, rf, c=1):
    y = ['SimilarityScore']
    y_train = train[y].values.ravel()
    X_train = train.drop(y, axis = 1)
    y_test = test[y]
    X_test = test.drop(y, axis = 1)

    # try this: OneVsRestClassifier(LinearSVC(penalty=rf, C=c, dual=rf == 'l2', max_iter=25000), n_jobs=-1)
    svc = LinearSVC(penalty=rf, C=c, dual=rf == 'l2', max_iter=25000)
    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    score = f1_score(y_test, predictions, average='weighted')
    print('Fold',str(fold_no),'F1 SCORE:', score)
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
    print("Average F1 SCORE of Support Vector Machine  is {:.2f}%".format(average / 10))


def optimize_c_parameter(train, test):
    X_train, y_train, X_test, y_test = train_test_init(train, test)

    # defining parameter range
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1]}

    # Refit an estimator using the best found parameters on the whole dataset.
    grid = GridSearchCV(LinearSVC(max_iter=25000), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train.values.ravel())

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


def support_vector_classifier(comments_data, processing_technique_applied):
    print(f"Support vector classifier {processing_technique_applied}")

    # Testing differences between regularisation functions
    print("> L1/L2 comparing")
    compare_regularisation_functions(comments_data, 'l1')
    compare_regularisation_functions(comments_data, 'l2')

    # Optimizing C parameter
    print("> Results with optimized C parameter")
    train, test = train_test_split(comments_data, test_size=0.2, random_state=42, shuffle=True)
    optimize_c_parameter(train, test)

