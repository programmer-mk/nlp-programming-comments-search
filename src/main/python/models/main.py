# imports
import numpy as np
import pandas as pd
from classifiers import SVC

RESOURCES_DIR = '../../resources/'
PROCESSED_DATA_DIR = 'processed_data'
RAW_DATA_DIR = 'raw_data'

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


def apply_all_classifiers(data):
    SVC.support_vector_classifier(data)

    # TODO: do next two classifiers @djojdanic @bselic
    #multinomialNB.multinomial_nb(data)
    #logistic_regression.logistic_regression(data)


def load_data():
    without_preprocessing_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/without_preprocessing.txt')
    lowercasing_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/lowercasing.txt')
    tf_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/tf.txt')
    tf_idf_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/tf_idf.txt')
    stemm_stopwords_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/stemming_and_remove_stopwords.txt')
    frequency_filtering_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/bigrams.txt')
    bigrams_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/bigrams.txt')
    trigrams_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/trigrams.txt')
    binary_bow_data = np.loadtxt(f'{RESOURCES_DIR}{PROCESSED_DATA_DIR}/binary_bow.txt')


def classifying():
    print("----------  No preprocessing(BOW) ----------")
    apply_all_classifiers("without_preprocessing.txt")

    print("----------  Lower casing ----------")
    apply_all_classifiers("lowercasing.txt")

    print("----------  Term Frequency ----------")
    apply_all_classifiers("tf.txt")

    #double check this
    #print("----------  Inverse Document Frequency ----------")
    #apply_all_classifiers("idf.txt")

    print("----------  Term Frequency–Inverse Document Frequency ----------")
    apply_all_classifiers("tf_idf.txt")

    print("----------  Stemming and stopwords ----------")
    apply_all_classifiers("stemming_and_remove_stopwords.txt")

    print("----------  Frequency word filtering ----------")
    apply_all_classifiers("frequency_filtering.txt")

    print("----------  Bigram preprocessing ----------")
    apply_all_classifiers("bigrams.txt")

    print("----------  Trigram preprocessing ----------")
    apply_all_classifiers("trigrams.txt")

    print("----------  Binary Bag of Words ----------")
    apply_all_classifiers("binary_bow.txt")


if __name__ == "__main__":
    option = int(input("Choose option? \n"
                       "0 - classifying \n"
                       "1 - calculate comment annotation similarity \n"))

    # TODO: add comment annotation similarity @djojdanic @bselic
    if option == 0:
        load_data()
        classifying()
    else:
        print('percentage annotation calculation!')
        #percentage_calc.percentage_calculator()