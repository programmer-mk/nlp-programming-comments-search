# imports
import sys

import pandas as pd
from classifiers import logistic_regression
from classifiers import naive_bayes
from classifiers import support_vector_machine

from src.main.python.models.ranking.ranking import start_ranking

sys.path.append("../data_preprocessing")
from preprocessing import preprocessing_data

RESOURCES_DIR = '../../resources'
PROCESSED_DATA_DIR = 'processed_data'

operating_system = sys.platform

if operating_system == 'win32':
    RESOURCES_DIR = 'src\main/resources'

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


def apply_all_classifiers(data, processing_technique_applied):
    naive_bayes.naive_bayes_classifier(data, processing_technique_applied)
    logistic_regression.logistic_regression_classifier(data, processing_technique_applied)
    support_vector_machine.support_vector_classifier(data, processing_technique_applied)


def initialize_data(all_data):
    global without_preprocessing_data, lowercasing_data, tf_data, tf_idf_data, stemm_stopwords_data, \
        frequency_filtering_data, bigrams_data, trigrams_data, binary_bow_data

    without_preprocessing_data = all_data[0]
    lowercasing_data = all_data[1]
    stemm_stopwords_data = all_data[2]
    bigrams_data = all_data[3]
    trigrams_data = all_data[4]
    tf_data = all_data[5]
    tf_idf_data = all_data[6]
    frequency_filtering_data = all_data[7]
    binary_bow_data = all_data[8]


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

    # #trigrams_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/trigrams.csv', sep='\t')

    print('start reading binary bow data....')
    binary_bow_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/binary_bow.csv', sep='\t')

    print('start reading bigrams data ...')
    bigrams_data = pd.read_csv(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/bigrams.csv', sep='\t')


def load_target_column():
    global data_target_column
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns, sep='\t')
    data_target_column = data_columns[['SimilarityScore']]
    list = ['0','1','2','3']
    data_target_column = data_target_column[data_target_column.SimilarityScore.isin(list)]
    print('end loading labels..')


def classifying():
    print("----------  No preprocessing(BOW) ----------")
    without_preprocessing_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(without_preprocessing_data, 'without processing')

    print("----------  Lower casing ----------")
    lowercasing_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(lowercasing_data, 'lowercasing')

    print("----------  Term Frequency ----------")
    tf_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(tf_data, 'term frequency')

    #double check this
    #print("----------  Inverse Document Frequency ----------")
    #apply_all_classifiers("idf.txt")

    print("----------  Term Frequency–Inverse Document Frequency ----------")
    tf_idf_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(tf_idf_data, 'TFIDF')

    print("----------  Stemming and stopwords ----------")
    stemm_stopwords_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(stemm_stopwords_data, 'stemming+stopwords')

    print("----------  Frequency word filtering ----------")
    frequency_filtering_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(frequency_filtering_data, 'frequency filtering')

    print("----------  Bigram preprocessing ----------")
    bigrams_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(bigrams_data, 'bigrams')

    # print("----------  Trigram preprocessing ----------")
    # trigrams_data['SimilarityScore'] = data_target_column
    # apply_all_classifiers(trigrams_data, 'trigrams')

    print("----------  Binary Bag of Words ----------")
    binary_bow_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(binary_bow_data, 'binary bow')


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

        if option >= 0 or option <= 2:
            load_target_column()
            if option == 0:
                print('Classifying started!')
                if load_from_disk:
                    # TODO: fix this if someone wants to read data from disk
                    preprocessed_data = load_preprocessed_data_from_disk()
                else:
                    preprocessed_data = preprocessing_data(False) # return one bag of words
                initialize_data(preprocessed_data)
                classifying()
            elif option == 1:
                print('Ranking started!')
                preprocessed_data = preprocessing_data(True) # return two bag of words

                '''
                    set index 0, for testing purposing of each method increase this index and comment other preprocessing methods
                    because of we don't want to load all data in RAM
                '''
                start_ranking(preprocessed_data[0][0], preprocessed_data[0][1])

            correct_input = True
        else:
            print('Incorrect input')
