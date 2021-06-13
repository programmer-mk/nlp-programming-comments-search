#from timeit import default_timer as timer
import threading
import time
from joblib import Parallel, delayed
import os
import pandas as pd
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup

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


"""
    merged_text param is used if we want to return vectorized full text( CommentText + QueryText)
    comment_of_query param is used if we want to return vectorized CommentText or QueryText (true -> CommentText, false -> QueryText). Default value is True
"""
def create_bag_of_words(data_set, merged_text, comment_or_query=True):
    cv_unigram = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_unigram.fit(data_set["Merged Text"])
    for index, value in enumerate(cv_unigram.get_feature_names()):
        print(f'{value}:{index}')
    if merged_text:
        bow = pd.DataFrame(cv_unigram.fit_transform(data_set["Merged Text"]).todense())
        return bow
    elif comment_or_query:
        bow = pd.DataFrame(cv_unigram.transform(data_set["CommentText"]).todense())
        return bow
    else:
        bow = pd.DataFrame(cv_unigram.transform(data_set["QueryText"]).todense())
        return bow


def get_stemming_result(file_name):
    return pd.read_csv(f"{MAIN_CONFIG_DIR}/{file_name}", names = ['CommentText', 'QueryText'], sep='\t', encoding='utf-8')


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


def prepare_files_for_stemming(data_frame, input_file_name):
    write_data_frame_to_file(data_frame, MAIN_CONFIG_DIR, input_file_name)


def do_file_stemming(data, input_file_name, output_file_name):
    prepare_files_for_stemming(data, input_file_name)
    execute_stemming_command(input_file_name, output_file_name)
    stemm_result = get_stemming_result(output_file_name)
    remove_files([f'{MAIN_CONFIG_DIR}/{input_file_name}', f'{MAIN_CONFIG_DIR}/{output_file_name}'])
    return stemm_result


def stemming_and_remove_stopwords(data_set, separate_query_and_comment_text):
    stopwords = read_stop_words()
    data_set['CommentText'] = data_set['CommentText'].apply(lambda text: remove_stop_words_and_tokenize_word(text, stopwords))
    data_set['QueryText'] = data_set['QueryText'].apply(lambda text: remove_stop_words_and_tokenize_word(text, stopwords))
    if separate_query_and_comment_text:
        stemmed_data = do_file_stemming(data_set, 'input-stemming.csv', 'output-stemming.csv')
        bow_comment = create_bag_of_words(stemmed_data, False, True)
        bow_query = create_bag_of_words(stemmed_data, False, False)
        return bow_comment, bow_query
    else:
        return create_bag_of_words(do_file_stemming(data_set, 'input-stemming.csv', 'output-stemming.csv'), True), None


"""
    This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary.
     (Set idf and normalization to False to get 0/1 outputs).
"""
def binary_bow(data_set, separate_query_and_comment_text):
    binary_tf = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False, binary=True)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    if separate_query_and_comment_text:
        binary_tf.fit(data_set["Merged Text"])
        bow_comment = pd.DataFrame(binary_tf.fit_transform(data_set["CommentText"]).todense())
        bow_query = pd.DataFrame(binary_tf.fit_transform(data_set["QueryText"]).todense())
        print(binary_tf.get_feature_names())
        return bow_comment, bow_query
    else:
        binary_tf.fit(data_set["Merged Text"])
        print(binary_tf.get_feature_names())
        bow = pd.DataFrame(binary_tf.fit_transform(data_set["Merged Text"]).todense())
        return bow, None


"""
    ignore terms that have a document frequency strictly higher than 0.9 and lower than 0.1
    parameters could be tuned if it's needed
"""
def frequency_filtering(data_set, separate_query_and_comment_text):
    freq_filter = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False,  min_df=0.03, max_df=0.97)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    if separate_query_and_comment_text:
        freq_filter.fit(data_set["Merged Text"])
        bow_comment = pd.DataFrame(freq_filter.transform(data_set["CommentText"]).todense())
        bow_query = pd.DataFrame(freq_filter.transform(data_set["QueryText"]).todense())
        print(freq_filter.get_feature_names())
        return bow_comment, bow_query
    else:
        freq_filter.fit(data_set["Merged Text"])
        print(freq_filter.get_feature_names())
        bow = pd.DataFrame(freq_filter.fit_transform(data_set["Merged Text"]).todense())
        return bow, None


def tf(data_set, separate_query_and_comment_text):
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=False, lowercase=False, analyzer='word') # this guy removes words with  only one character
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    if separate_query_and_comment_text:
        tf_vectorizer.fit(data_set["Merged Text"])
        tfs_comment = tf_vectorizer.transform(data_set["CommentText"])
        tfs_query = tf_vectorizer.transform(data_set["QueryText"])
        return pd.DataFrame(tfs_comment.toarray()), pd.DataFrame(tfs_query.toarray())
    else:
        tfs = tf_vectorizer.fit_transform(data_set["Merged Text"])
        pda = pd.DataFrame(tfs.toarray())
        return pda, None


def tf_idf(data_set, separate_query_and_comment_text):
    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, lowercase=False, analyzer='word') # this guy removes words with  only one character
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]

    if separate_query_and_comment_text:
        tf_idf_vectorizer.fit(data_set["Merged Text"])
        tf_idfs_comment = tf_idf_vectorizer.transform(data_set["CommentText"])
        tf_idfs_query = tf_idf_vectorizer.transform(data_set["QueryText"])
        print(tf_idf_vectorizer.get_feature_names())
        return pd.DataFrame(tf_idfs_comment.toarray()) , pd.DataFrame(tf_idfs_query.toarray())
    else:
        tf_idfs = tf_idf_vectorizer.fit_transform(data_set["Merged Text"])
        print(tf_idf_vectorizer.get_feature_names())
        pda = pd.DataFrame(tf_idfs.toarray())
        return pda, None


def bigrams(data_set, separate_query_and_comment_text):
    cv_bigram = CountVectorizer(ngram_range=(2, 2), lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    if separate_query_and_comment_text:
        cv_bigram.fit(data_set["Merged Text"])
        bow_comment = pd.DataFrame(cv_bigram.transform(data_set["CommentText"]).todense())
        bow_query = pd.DataFrame(cv_bigram.transform(data_set["QueryText"]).todense())
        print(cv_bigram.get_feature_names())
        return bow_comment, bow_query

    else:
        bow = pd.DataFrame(cv_bigram.fit_transform(data_set["Merged Text"]).todense())
        print(cv_bigram.get_feature_names())
        return bow, None


def trigrams(data_set, separate_query_and_comment_text):
    cv_trigram = CountVectorizer(ngram_range=(3, 3), lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    if separate_query_and_comment_text:
        cv_trigram.fit(data_set["Merged Text"])
        bow_comment = pd.DataFrame(cv_trigram.transform(data_set["CommentText"]).todense())
        bow_query = pd.DataFrame(cv_trigram.transform(data_set["QueryText"]).todense())
        print(cv_trigram.get_feature_names())
        return bow_comment, bow_query

    else:
        cv_trigram.fit(data_set["Merged Text"])
        print(cv_trigram.get_feature_names())
        bow = pd.DataFrame(cv_trigram.fit_transform(data_set["Merged Text"]).todense())
        return bow, None


def without_preprocessing(data_set, separate_query_and_comment_text):
    if separate_query_and_comment_text:
        return create_bag_of_words(data_set, False), create_bag_of_words(data_set, False, False)
    else:
        return create_bag_of_words(data_set, True), None


def lowercasing(data_set, separate_query_and_comment_text):
    data_set['CommentText'] = data_set['CommentText'].apply(lambda comment: comment.lower())
    data_set['QueryText'] = data_set['QueryText'].apply(lambda comment: comment.lower())
    if separate_query_and_comment_text:
        return create_bag_of_words(data_set, False), create_bag_of_words(data_set, False, False)
    else:
        return create_bag_of_words(data_set, True), None


processing_steps = {
    0: without_preprocessing,
    1: lowercasing,
    2: stemming_and_remove_stopwords,
    3: bigrams,
    4: trigrams,
    5: tf,
    #6: idf, skipping for now, not sure that make sense doing it
    6: tf_idf,
    7: frequency_filtering,
    8: binary_bow
}


def read_raw_data():
    columns = ['ProgrammingLanguage', 'QueryId','PairID', 'QueryText', 'CommentText','SimilarityScore']
    comments = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", sep = "\t", names=columns)
    comments.drop(index=comments.index[0], axis=0, inplace=True)
    return comments[['QueryText', 'CommentText']]


def remove_outliers(data):
    # TODO: implement outliers removal logic
    return data


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def start_processing(step, data_set, save_to_disk, separate_query_and_comment_text):
    processing_technique = processing_steps.get(step)
    if processing_technique is None:
        print('Unrecognized processing type. Please specify number between 1 - x')
        return None, None
    else:
        processed_data = processing_technique(data_set, separate_query_and_comment_text)
        print(processed_data)
        if save_to_disk:
            write_bow_data_frame_to_file(processed_data, f'{PROCESSED_DATA_DIR}/{processing_technique.__name__}.csv')
            return None, None
        else:
            return processed_data


def init_preprocessing(data, lower_case = False, drop_na = True, remove_html_tags = True, remove_special_chars = True, remove_extra_whitespace = True):

    output_data = data.copy()

    # Remove special characters, outliers, duplicates, null values, html tags

    # Remove null values
    if drop_na:
        output_data.dropna(how='any', inplace=True)

    # Remove html tags
    if remove_html_tags:
        output_data['CommentText'] = output_data['CommentText'].apply(lambda text: strip_html_tags(text))
        output_data['QueryText'] = output_data['QueryText'].apply(lambda text: strip_html_tags(text))

    # Lowercase
    if lower_case:
        output_data['CommentText'] = output_data['CommentText'].apply(lambda text: text.lower())
        output_data['QueryText'] = output_data['QueryText'].apply(lambda text: text.lower())

    # Remove special characters
    if remove_special_chars:
        output_data['CommentText'] = output_data['CommentText'].apply(lambda text: remove_special_characters(text, remove_digits=True))
        output_data['QueryText'] = output_data['QueryText'].apply(lambda text: remove_special_characters(text, remove_digits=True))

    # Remove extra whitespace
    if remove_extra_whitespace:
        output_data['CommentText'] = output_data['CommentText'].apply(lambda text: re.sub(' +',' ', text))
        output_data['QueryText'] = output_data['QueryText'].apply(lambda text:re.sub(' +',' ', text))

    return output_data


"""
    :param separate_query_and_comment_text
    used to create separate bag of words for QueryText and CommentText, this is need for ranking if it's enabled(true)
    if it's disabled then processing is used for classifying so one bag of words is needed in that case(for merged CommentText and QueryText)
"""
def preprocessing_data(separate_query_and_comment_text):
    all_preprocessed_data = []
    raw_data = read_raw_data()
    cleaned_data = init_preprocessing(raw_data)
    for step in list(range(9)):
        if separate_query_and_comment_text:
            preprocessed_data_comments, preprocessed_data_queries = start_processing(step, cleaned_data.copy(), False, separate_query_and_comment_text)
            all_preprocessed_data.append((preprocessed_data_comments, preprocessed_data_queries))
        else:
            preprocessed_data, _ = start_processing(step, cleaned_data.copy(), False, separate_query_and_comment_text)
            all_preprocessed_data.append((preprocessed_data, None))
    return all_preprocessed_data


if __name__ == '__main__':
    raw_data = read_raw_data()
    preprocessed_data = init_preprocessing(raw_data.copy())
    for step in list(range(9)):
        start_processing(step, preprocessed_data.copy(), True, False)