import os
import pandas as pd
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.toktok import ToktokTokenizer


MAIN_CONFIG_DIR = '../../config'
PROCESSED_DATA_DIR = '../../resources/processed_data'
RESOURCES_DIR = '../../resources'
tokenizer = ToktokTokenizer()

operating_system = sys.platform
if operating_system == 'win32':
    MAIN_CONFIG_DIR = 'src\main/config'
    PROCESSED_DATA_DIR = 'src\main/resources/processed_data'
    RESOURCES_DIR = 'src\main/resources'


def remove_files(files):
    for file in files:
        os.remove(file)


def remove_stop_words_and_tokenize_word(text, stopwords):
    words = tokenizer.tokenize(text)
    return "".join([word + " " for word in words if word not in stopwords])


def read_stop_words():
    stopwords = []
    with open(f'{MAIN_CONFIG_DIR}/stopwords.txt', 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            # remove linebreak
            line_cleaned = line[:-1]
            stopwords.append(line_cleaned)
    return stopwords


def create_bag_of_words(data_set):
    cv_unigram = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_unigram.fit(data_set["Merged Text"])
    print(cv_unigram.get_feature_names())
    bow = pd.DataFrame(cv_unigram.fit_transform(data_set["Merged Text"]).todense())
    return bow


def get_stemming_result(file_name):
    return pd.read_csv(f"{MAIN_CONFIG_DIR}/{file_name}", names = ['QueryText', 'CommentText'], sep='\t', encoding='utf-8')


def execute_stemming_command(input_file_name, output_file_name):
    """
        StemmerIds:

    1 - Keselj & Sipka - Greedy
    2 - Keselj & Sipka - Optimal
    3 - Milosevic
    4 - Ljubesic & Pandzic
    """

    stemmer_id = 4
    stemmer_implementation = f'{MAIN_CONFIG_DIR}/SCStemmers.jar'
    stem_command = f'java -jar {stemmer_implementation} {stemmer_id} {MAIN_CONFIG_DIR}/{input_file_name} {MAIN_CONFIG_DIR}/{output_file_name}'
    os.system(stem_command)


def write_data_frame_to_file(data_frame, file_path, file_name):
    file_location = f'{file_path}/{file_name}'
    data_frame.to_csv(file_location , index=False, header=False, sep='\t', encoding='utf-8')


def write_bow_data_frame_to_file(data_frame, file_path):
    data_frame.to_csv(f'{file_path}', sep = '\t', index = False)
    #np.savetxt(f'{file_path}', data_frame.to_numpy(), fmt="%f")


def prepare_files_for_stemming(data_frame, input_file_name):
    write_data_frame_to_file(data_frame, MAIN_CONFIG_DIR, input_file_name)


def do_file_stemming(data, input_file_name, output_file_name):
    prepare_files_for_stemming(data, input_file_name)
    execute_stemming_command(input_file_name, output_file_name)
    stemm_result = get_stemming_result(output_file_name)
    remove_files([f'{MAIN_CONFIG_DIR}/{input_file_name}', f'{MAIN_CONFIG_DIR}/{output_file_name}'])
    return stemm_result

def stemming_and_remove_stopwords(data_set):
    stopwords = read_stop_words()
    data_set['CommentText'] = data_set['CommentText'].apply(lambda text: remove_stop_words_and_tokenize_word(text, stopwords))
    data_set['QueryText'] = data_set['QueryText'].apply(lambda text: remove_stop_words_and_tokenize_word(text, stopwords))
    return create_bag_of_words(do_file_stemming(data_set, 'input-stemming.csv', 'output-stemming.csv'))


"""
    This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary.
     (Set idf and normalization to False to get 0/1 outputs).
"""
def binary_bow(data_set):
    binary_tf = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False, binary=True)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    binary_tf.fit(data_set["Merged Text"])
    print(binary_tf.get_feature_names())
    bow = pd.DataFrame(binary_tf.fit_transform(data_set["Merged Text"]).todense())
    return bow


"""
    ignore terms that have a document frequency strictly higher than 0.9 and lower than 0.1
    parameters could be tuned if it's needed
"""
def frequency_filtering(data_set):
    freq_filter = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False,  min_df=0.1, max_df=0.9)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    freq_filter.fit(data_set["Merged Text"])
    print(freq_filter.get_feature_names())
    bow = pd.DataFrame(freq_filter.fit_transform(data_set["Merged Text"]).todense())
    return bow


def idf(data_set):
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv = CountVectorizer(lowercasing )
    words = cv.fit_transform(data_set["Merged Text"])
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(words)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names(), 'idf_weights':tfidf_transformer.idf_})
    return idf


def tf(data_set):
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=False, lowercase=False, analyzer='word') # this guy removes words with  only one character
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    # tfs = pd.DataFrame(tf_vectorizer.fit_transform(data_set["Merged Text"]).todense())
    tfs = tf_vectorizer.fit_transform(data_set["Merged Text"])
    pda = pd.DataFrame(tfs.toarray())
    return pda


def tf_idf(data_set):
    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, lowercase=False, analyzer='word') # this guy removes words with  only one character
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    tf_idfs = tf_idf_vectorizer.fit_transform(data_set["Merged Text"])
    print(tf_idf_vectorizer.get_feature_names())
    #df_tf_idf = pd.DataFrame(tf_idfs, index=tf_idf_vectorizer.get_feature_names(),columns=["tf_idf_weights"])
    pda = pd.DataFrame(tf_idfs.toarray())
    # return pd.DataFrame(pda)
    return pda


def bigrams(data_set):
    cv_bigram = CountVectorizer(ngram_range=(2, 2), lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_bigram.fit(data_set["Merged Text"])
    print(cv_bigram.get_feature_names())
    bow = pd.DataFrame(cv_bigram.fit_transform(data_set["Merged Text"]).todense())
    return bow


def trigrams(data_set):
    cv_trigram = CountVectorizer(ngram_range=(3, 3), lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_trigram.fit(data_set["Merged Text"])
    print(cv_trigram.get_feature_names())
    bow = pd.DataFrame(cv_trigram.fit_transform(data_set["Merged Text"]).todense())
    return bow


def without_preprocessing(data_set):
    return create_bag_of_words(data_set)


def lowercasing(data_set):
    data_set['CommentText'] = data_set['CommentText'].apply(lambda comment: comment.lower())
    data_set['QueryText'] = data_set['QueryText'].apply(lambda comment: comment.lower())
    return create_bag_of_words(data_set)


processing_steps = {
    0: without_preprocessing,
    1: lowercasing,
    2: stemming_and_remove_stopwords,
    3: bigrams,
    #4: trigrams,
    5: tf,
    #6: idf, skipping for now, not sure that make sense doing it
    6: tf_idf,
    7: frequency_filtering,
    8: binary_bow

}


def read_raw_data():
    columns = ['ProgrammingLanguage', 'QueryId','PairID', 'QueryText', 'CommentText','SimilarityScore']
    comments = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", sep = "\t", names=columns)
    return comments[['QueryText', 'CommentText']]


def remove_outliers(data):
    # TODO: implement outliers removal logic
    return data


def start_processing(step, data_set):
    processing_technique = processing_steps.get(step)
    if processing_technique is None:
        print('Unrecognized processing type. Please specify number between 1 - x')
    else:
        processed_data = processing_technique(data_set)
        print(processed_data)
        write_bow_data_frame_to_file(processed_data, f'{PROCESSED_DATA_DIR}/{processing_technique.__name__}.csv')


def init_preprocessing(data):
    # Remove special characters, outliers, duplicates, null values, html tags
    data.dropna(how='any', inplace=True)
    return data


if __name__ == '__main__':
    for step in list(range(9)):
        raw_data = read_raw_data()
        preprocessed_data = init_preprocessing(raw_data)
        start_processing(step, preprocessed_data)