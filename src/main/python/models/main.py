# imports
import sys
print(sys.path)

import pandas as pd
from classifiers import logistic_regression
from classifiers import naive_bayes
from classifiers import support_vector_machine
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# from src.main.python.models.ranking.ranking import start_ranking
from ranking.ranking import start_ranking

sys.path.append("../data_preprocessing")
# from classifiers.preprocessing_copy import preprocessing_data,bigrams,trigrams,tf_idf,frequency_filtering,freq_filter,tf_idf_vectorizer,cv_bigram,cv_trigram
# from classifiers.preprocessing_copy import *
from preprocessing import *
RESOURCES_DIR = '../../resources'

PROCESSED_DATA_DIR = f'{RESOURCES_DIR}/processed_data'
operating_system = sys.platform

if operating_system == 'win64':
    RESOURCES_DIR = 'src\main/resources'
    PROCESSED_DATA_DIR = f'{RESOURCES_DIR}/processed_data'


raw_data = None
without_preprocessing_data = None
lowercasing_data = None
tf_data = None
tf_idf_data = None
stemm_stopwords_data = None
frequency_filtering_data = None
bigrams_data = None
trigrams_data = None
binary_bow_data = None

# SimilarityScore
data_target_column = None


def write_test_results_to_file(processing_technique, test_score, predictions, y_test, model_type):
    f = open(f"../../resources/classification-results/{processing_technique}/{processing_technique}-test-results-{model_type}.txt", "a")
    f.write(f"Test score for {processing_technique}= {test_score}")
    f.write("\n")
    f.write(f"{confusion_matrix(y_test, predictions, labels=['0','1','2','3'])}")
    f.close()


def prepare_test_data(test,processing_technique_applied, vectorizer):

    if processing_technique_applied == 'TF-IDF':
        test["Merged Text"] = test["CommentText"] + ' ' + test["QueryText"]
        test_preprocessed = pd.DataFrame(vectorizer.transform(test["Merged Text"]).toarray())

    elif processing_technique_applied == 'frequency-filtering':
        test["Merged Text"] = test["CommentText"] + ' ' + test["QueryText"]
        test_preprocessed = pd.DataFrame(vectorizer.transform(test["Merged Text"]).todense())

    elif processing_technique_applied == 'bigrams':
        test["Merged Text"] = test["CommentText"] + ' ' + test["QueryText"]
        test_preprocessed = pd.DataFrame(vectorizer.transform(test["Merged Text"]).todense())

    elif processing_technique_applied == 'trigrams':
        test["Merged Text"] = test["CommentText"] + ' ' + test["QueryText"]
        test_preprocessed = pd.DataFrame(vectorizer.transform(test["Merged Text"]).todense())

    else:
        test_preprocessed = test

    return test_preprocessed


def apply_all_classifiers(data, processing_technique_applied):
    
    train, test = train_test_split(data, test_size=0.05, random_state=42, shuffle=True)
    y_test = test['SimilarityScore']
    test = test.drop(['SimilarityScore'], axis=1)

    # NAIVE BAYES

    # mnb, vectorizer = naive_bayes.naive_bayes_classifier(train, processing_technique_applied)
    # test_preprocessed = prepare_test_data(test, processing_technique_applied, vectorizer)
    # predictions_nb = mnb.predict(test_preprocessed)
    # test_score_nb = f1_score(y_test, predictions_nb, average='weighted')
    # print(f'Done calculating predictions and f1_score - naive bayes - {processing_technique_applied}')
    # write_test_results_to_file(processing_technique_applied, test_score_nb, predictions_nb, y_test, "nb")
    # print(f'Done writing test results to file - naive bayes - {processing_technique_applied}')

    # SVM

    # optimal_svc, vectorizer = support_vector_machine.support_vector_classifier(train, processing_technique_applied)
    # test_preprocessed = prepare_test_data(test, processing_technique_applied, vectorizer)
    # predictions_svc = optimal_svc.predict(test_preprocessed)
    # test_score_svc = f1_score(y_test, predictions_svc, average='weighted')
    # print(f'Done calculating predictions and f1_score - svc - {processing_technique_applied}')
    # write_test_results_to_file(processing_technique_applied, test_score_svc, predictions_svc, y_test, "svm")
    # print(f'Done writing test results to file - svc - {processing_technique_applied}')

    # LOGISTIC REGRESSION

    optimal_lg, vectorizer, _optimal_c = logistic_regression.logistic_regression_classifier(train, processing_technique_applied)
    test_preprocessed = prepare_test_data(test, processing_technique_applied, vectorizer)
    predictions_lg = optimal_lg.predict(test_preprocessed)
    test_score_lg = f1_score(y_test, predictions_lg, average='weighted')
    print(f'Done calculating predictions and f1_score - lg - {processing_technique_applied}')
    write_test_results_to_file(processing_technique_applied, test_score_lg, predictions_lg, y_test, "lg")
    print(f'Done writing test results to file - lg - {processing_technique_applied}')
    print(f'Optimal c parameter: {_optimal_c}')


def initialize_data(all_data):
    global without_preprocessing_data, lowercasing_data, tf_data, tf_idf_data, stemm_stopwords_data, \
        frequency_filtering_data, bigrams_data, trigrams_data, binary_bow_data

    without_preprocessing_data = all_data[0][0]
    lowercasing_data = all_data[1][0]
    stemm_stopwords_data = all_data[2][0]
    bigrams_data = all_data[3][0]
    trigrams_data = all_data[4][0]
    tf_data = all_data[5][0]
    tf_idf_data = all_data[6][0]
    frequency_filtering_data = all_data[7][0]
    binary_bow_data = all_data[8][0]


def load_preprocessed_data_from_disk():
    global without_preprocessing_data, lowercasing_data, tf_data, tf_idf_data, stemm_stopwords_data,\
        frequency_filtering_data, bigrams_data, trigrams_data, binary_bow_data

    print('start reading without processing data....')
    without_preprocessing_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/without_preprocessing.csv', sep='\t')

    print('start reading lowercasing data....')
    lowercasing_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/lowercasing.csv', sep='\t')

    print('start reading tf data....')
    tf_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/tf.csv', sep='\t')

    print('start reading tfidf data....')
    tf_idf_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/tf_idf.csv', sep='\t')

    print('start reading stem + stopwords data....')
    stemm_stopwords_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/stemming_and_remove_stopwords.csv', sep='\t')

    print('start reading frequency filter data ....')
    frequency_filtering_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/frequency_filtering.csv', sep='\t')

    print('start reading trigrams data ....')
    trigrams_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/trigrams.csv', sep='\t')

    print('start reading binary bow data....')
    binary_bow_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/binary_bow.csv', sep='\t')

    print('start reading bigrams data ...')
    bigrams_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/bigrams.csv', sep='\t')


def load_target_column():
    global data_target_column
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns, sep='\t')
    data_target_column = data_columns[['SimilarityScore']]
    grades = ['0', '1', '2', '3']
    data_target_column = data_target_column[data_target_column.SimilarityScore.isin(grades)]
    print('end loading labels..')


def classifying():

    # print("----------  No preprocessing(BOW) ----------")
    # if not os.path.exists('../../resources/classification-results/without-processing'):
    #     os.mkdir('../../resources/classification-results/without-processing')
    # without_preprocessing_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(without_preprocessing_data, 'without-processing')
    #
    # print("----------  Lower casing ----------")
    # if not os.path.exists('../../resources/classification-results/lowercasing'):
    #     os.mkdir('../../resources/classification-results/lowercasing')
    # lowercasing_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(lowercasing_data, 'lowercasing')
    #
    # print("----------  Stemming and stopwords ----------")
    # if not os.path.exists('../../resources/classification-results/stemming+stopwords'):
    #     os.mkdir('../../resources/classification-results/stemming+stopwords')
    # stemm_stopwords_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(stemm_stopwords_data, 'stemming+stopwords')
    #
    # print("----------  Bigram preprocessing ----------")
    # if not os.path.exists('../../resources/classification-results/bigrams'):
    #     os.mkdir('../../resources/classification-results/bigrams')
    # bigrams_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(bigrams_data, 'bigrams')
    #
    print("----------  Trigram preprocessing ----------")
    if not os.path.exists('../../resources/classification-results/trigrams'):
        os.mkdir('../../resources/classification-results/trigrams')
    trigrams_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(trigrams_data, 'trigrams')
    #
    # print("----------  Term Frequency ----------")
    # if not os.path.exists('../../resources/classification-results/term-frequency'):
    #     os.mkdir('../../resources/classification-results/term-frequency')
    # tf_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(tf_data, 'term-frequency')

    # print("----------  Term Frequency–Inverse Document Frequency ----------")
    # if not os.path.exists('../../resources/classification-results/TF-IDF'):
    #     os.mkdir('../../resources/classification-results/TF-IDF')
    # tf_idf_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(tf_idf_data, 'TF-IDF')

    # print("----------  Frequency word filtering ----------")
    # if not os.path.exists('../../resources/classification-results/frequency-filtering'):
    #     os.mkdir('../../resources/classification-results/frequency-filtering')
    # frequency_filtering_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(frequency_filtering_data, 'frequency-filtering')
    #
    # print("----------  Binary Bag of Words ----------")
    # if not os.path.exists('../../resources/classification-results/binary-bow'):
    #     os.mkdir('../../resources/classification-results/binary-bow')
    # binary_bow_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(binary_bow_data, 'binary-bow')


if __name__ == "__main__":

    correct_input = False

    while not correct_input:
        menu_message = "Choose option? \n"\
                        "0 - classifying \n"\
                        "1 - ranking \n"\
                        "2 - exit \n"

        option = int(input(menu_message))
        # data should be preprocessed and saved to disk
        # with this enabled, reading from disk will take time
        load_from_disk = False

        if 0 <= option <= 2:
            load_target_column()
            if option == 0:
                print('Classifying started!')
                if load_from_disk:
                    # TODO: fix this if someone wants to read data from disk
                    preprocessed_data = load_preprocessed_data_from_disk()
                else:
                    preprocessed_data = preprocessing_data(False)  # return one bag of words
                initialize_data(preprocessed_data)
                classifying()
            elif option == 1:
                print('Ranking started!')
                preprocessed_data = preprocessing_data(True)  # return two bag of words

                '''
                    set index 0, for testing purposing of each method increase this index and comment other preprocessing methods
                    because of we don't want to load all data in RAM
                '''
                start_ranking(preprocessed_data[0][0], preprocessed_data[0][1])

            correct_input = True
        else:
            print('Incorrect input')
