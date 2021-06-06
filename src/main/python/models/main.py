# imports
import numpy as np
import pandas as pd
from .classifiers import SVC
from .classifiers import naive_bayes
import sys

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


def apply_all_classifiers(data):
    #SVC.support_vector_classifier(data)
    naive_bayes.naive_bayes_classifier(data)

    # TODO: do next two classifiers @djojdanic @bselic
    #multinomialNB.multinomial_nb(data)
    #logistic_regression.logistic_regression(data)


def load_data():
    global without_preprocessing_data, lowercasing_data, tf_data, tf_idf_data, stemm_stopwords_data,\
        frequency_filtering_data, bigrams_data, trigrams_data, binary_bow_data

    without_preprocessing_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/without_preprocessing.txt'))
    lowercasing_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/lowercasing.txt'))
    tf_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/tf.txt'))
    tf_idf_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/tf_idf.txt'))
    stemm_stopwords_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/stemming_and_remove_stopwords.txt'))
    frequency_filtering_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/frequency_filtering.txt'))
    bigrams_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/bigrams.txt'))
    trigrams_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/trigrams.txt'))
    binary_bow_data = pd.DataFrame(np.loadtxt(f'{RESOURCES_DIR}/{PROCESSED_DATA_DIR}/binary_bow.txt'))


def load_target_column():
    global data_target_column
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns)
    data_target_column = data_columns[['SimilarityScore']]                      


def classifying():
    print("----------  No preprocessing(BOW) ----------")
    without_preprocessing_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(without_preprocessing_data)

    print("----------  Lower casing ----------")
    lowercasing_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(lowercasing_data)

    print("----------  Term Frequency ----------")
    apply_all_classifiers(tf_data)

    #double check this
    #print("----------  Inverse Document Frequency ----------")
    #apply_all_classifiers("idf.txt")

    print("----------  Term Frequency–Inverse Document Frequency ----------")
    tf_idf_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(tf_idf_data)

    print("----------  Stemming and stopwords ----------")
    stemm_stopwords_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(stemm_stopwords_data)

    print("----------  Frequency word filtering ----------")
    frequency_filtering_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(frequency_filtering_data)

    print("----------  Bigram preprocessing ----------")
    bigrams_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(bigrams_data)

    print("----------  Trigram preprocessing ----------")
    trigrams_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(trigrams_data)

    print("----------  Binary Bag of Words ----------")
    binary_bow_data['SimilarityScore'] = data_target_column
    apply_all_classifiers(binary_bow_data)


if __name__ == "__main__":

    correct_input = False

    while correct_input == False:
        menu_message = "Choose option? \n"\
                        "0 - classifying \n"\
                        "1 - calculate comment annotation similarity \n"\
                        "2 - exit \n"

        option = int(input(menu_message))

        if option < 0 or option > 2:
            # TODO: add comment annotation similarity @djojdanic @bselic
            if option == 0:
                load_target_column()
                load_data()
                classifying()
            elif option == 1:
                print('percentage annotation calculation!')
                #percentage_calc.percentage_calculator()
            correct_input = True
        else:
            print()
            print('Incorrect input')        