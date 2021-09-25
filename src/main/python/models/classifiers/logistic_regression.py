from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .train_helper import train_test_init
from imblearn.over_sampling import SMOTE
import pandas as pd
import sys

sys.path.append("../data_preprocessing")
from .preprocessing import tf_idf, frequency_filtering, bigrams, trigrams
import operator

PROCESSED_DATA_DIR = '../../resources/classification-results'

operating_system = sys.platform
if operating_system == 'win32':
    PROCESSED_DATA_DIR = 'src\main/resources/classification-results'

def train_model(train, test, fold_no, rf, processing_technique, c_param=0.01):
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
        '3': weights[3] * 12
    }

    print(f'weights are: {weights}')
    if rf == "l1":
        solver = 'liblinear'
        logistic_regression = LogisticRegression(penalty=rf, C=c_param, solver = solver, max_iter=200,verbose=10, random_state=42 , multi_class='ovr', n_jobs=-1, class_weight=class_weights)
    else:
        solver = 'lbfgs'
        logistic_regression = LogisticRegression(penalty='l2', random_state=42, C=c_param, n_jobs=-1, solver=solver, multi_class='ovr',
                                 max_iter=80,verbose=10, class_weight='balanced')

    logistic_regression.fit(X_train,y_train)
    predictions = logistic_regression.predict(X_test)
    print(confusion_matrix(y_test, predictions, labels=['0','1','2','3']))
    score = f1_score(y_test, predictions, average='weighted')

    f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-{c_param}-{rf}-{fold_no}-lg.txt", "a")
    f.write("\n")
    f.write(f"'Fold',{str(fold_no)},'F1 SCORE:',{score}")
    f.write("\n")
    f.write(f"{confusion_matrix(y_test, predictions, labels=['0','1','2','3'])}")
    f.close()

    print('Fold',str(fold_no),'F1 SCORE:', score)
    return score

def find_optimal_c_parameter(data_frame, r_type, processing_technique):

    data_frame.dropna(how='any', inplace=True)
    skf = StratifiedKFold(n_splits=10)
    target = data_frame.loc[:,'SimilarityScore']

    fold_no = 1
    average = 0

    f1_results = {}

    # defining c parameter range
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1]}

    global optimal_svc, svc
    max_f1_result = 0

    c_param_values = list(param_grid.values())[0]
    for c_parameter in c_param_values:   

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

            score, svc = train_model(train_preprocessed, test_preprocessed, fold_no, r_type, processing_technique, int(c_parameter))
            
            average += score
            fold_no += 1
    
        f1_result = average / 10
        f1_results[c_parameter] = f1_result

        if f1_result > max_f1_result:
                max_f1_result = f1_result
                optimal_svc = svc

        average = 0
        fold_no = 1

        print("Average F1 SCORE of Logistic Regression (" + r_type + ") is {:.2f}%".format(f1_result))
        f = open(f"{PROCESSED_DATA_DIR}/{processing_technique}/{processing_technique}-{c_parameter}-{r_type}-fold-average-lg.txt", "a")
        f.write("\n")
        f.write("Average F1 SCORE of Logistic Regression (" + r_type + ") is {:.2f}%".format(f1_result))
        f.write("\n")
        f.close()

    optimal_c_parameter = max(f1_results.items(), key=operator.itemgetter(1))[0]

    return optimal_c_parameter, max_f1_result, optimal_svc

def logistic_regression_classifier(comments_data, processing_technique_applied):
    print(f"Logistic regression classifier - Processing technique applied: {processing_technique_applied}")

    # train, test = train_test_split(comments_data, test_size=0.05, random_state=42, shuffle=True)

    # parametri uticu na model; parametri mnoze odlike i definisu ravan
    # C parametar raste -> manja regularizacija
    # C parametar blizu nule -> beskonacna regularizacija - losa stvar - parametri ne mogu da
    # dobiju zeljene vrednosti jer im parametar C to ne dozvoljava

    # Testing differences between regularisation functions
    print("Finding an optimal C parameter and comparing max f1 result for l1 and l2...")
    optimal_c_parameter_for_l1, max_f1_result_for_l1, optimal_lg_l1 = find_optimal_c_parameter(comments_data, 'l1', processing_technique_applied) 
    #parametri po modulu
    optimal_c_parameter_for_l2, max_f1_result_for_l2, optimal_lg_l2 = find_optimal_c_parameter(comments_data, 'l2', processing_technique_applied) 
    #parametri na kvadrat

    optimal_lg = optimal_lg_l1
    print('\n')
    if max_f1_result_for_l1 > max_f1_result_for_l2:
        print(f"l1 has better f1 result; l1 = {max_f1_result_for_l1}, l2 = {max_f1_result_for_l2}")
    elif max_f1_result_for_l1 < max_f1_result_for_l2:
        print(f"l2 has better f1 result; l1 = {max_f1_result_for_l1}, l2 = {max_f1_result_for_l2}")
        optimal_lg = optimal_lg_l2
    else:
        print(f"l1 and l2 have the same f1 result; l1 = l2 = {max_f1_result_for_l2}")

    return optimal_lg

    