import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

MAIN_CONFIG_DIR = '../../config'

cv_unigram = CountVectorizer(ngram_range=(1, 1))
cv_bigram = CountVectorizer(ngram_range=(2, 2))
cv_trigram = CountVectorizer(ngram_range=(3, 3))


def remove_stop_words_and_tokenize_word(text, stopwords):
    words = word_tokenize(text)
    return "".join([word + " " for word in words if word not in stopwords.words()])


def read_stop_words():
    stopwords = []
    with open(f'{MAIN_CONFIG_DIR}/stopwords.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            # remove linebreak
            line_cleaned = line[:-1]
            stopwords.append(line_cleaned)
    return stopwords


def get_stemming_result(file_name):
    stemmed_corpus = []
    with open(f'{MAIN_CONFIG_DIR}/{file_name}', 'w') as file:
        lines = file.readlines()
        for line in lines:
            # remove linebreak
            line_cleaned = line[:-1]
            stemmed_corpus.append(line_cleaned)
    return pd.Series(stemmed_corpus)


def execute_stemming_command(input_file_name, output_file_name):
    """
        StemmerIds:

    1 - Keselj & Sipka - Greedy
    2 - Keselj & Sipka - Optimal
    3 - Milosevic
    4 - Ljubesic & Pandzic
    """

    stemmer_id = 4
    stemmer_implementation = 'SCStemmers.jar'
    stem_command = f'java -jar {stemmer_implementation} {stemmer_id} {input_file_name} {output_file_name}'
    os.system(stem_command)


def prepare_files_for_stemming(data_frame, input_file_name):
    with open(f'{MAIN_CONFIG_DIR}/{input_file_name}', 'w') as file:
        for row in data_frame:
            file.write(row)
            file.write('\n')


def do_file_stemming(input_file_name, output_file_name):
    prepare_files_for_stemming(input_file_name, output_file_name)
    execute_stemming_command(input_file_name, output_file_name)
    return get_stemming_result(output_file_name)


def stemming_and_remove_stopwords(data_set):
    stopwords = read_stop_words()
    data_set['CommentText'] = do_file_stemming('input-stemming.txt', 'input-stemming.txt')
    data_set['CommentText'] = data_set['CommentText'].apply(remove_stop_words_and_tokenize_word, stopwords)
    return create_bag_of_words(data_set)


def without_preprocessing(data_set):
    return create_bag_of_words(data_set)


def lowercasing(data_set):
    data_set['CommentText'] = data_set['CommentText'].apply(lambda comment: comment.lower())
    return create_bag_of_words(data_set)


processing_steps = {
    1: without_preprocessing,
    2: lowercasing,
    3: stemming_and_remove_stopwords,
}


def read_raw_data():
    comments = pd.read_csv("../resources/comments.txt", sep="\t", names=['ProgrammingLanguageName', 'QueryID',
                                                                         'PairID', 'QueryText', 'CommentText',
                                                                         'SimilarityScore'])
    return comments[['CommentText', 'SimilarityScore']]


def init_variables(comments):
    unigram_counts = cv_unigram.fit_transform(comments['CommentText'])
    cv_bigram.fit(comments['CommentText'])
    cv_trigram.fit(comments['CommentText'])


def remove_outliers(data):
    return data


def create_bag_of_words(data_set):
    return cv_unigram.transform(data_set['CommentText'])


def start_processing(step, data_set):
    processing_technique = processing_steps.get(step)
    if processing_technique is None:
        print('Unrecognized processing type. Please specify number between 1 - x')
    else:
        processing_technique(data_set)


def remove_special_characters(text):
    # removes special characters with ' '
    cleaned = re.sub('[^a-zA-z\s]', ' ', text)
    cleaned = re.sub('_', ' ', cleaned)

    # Change any white space and new line to one space
    cleaned = cleaned.replace('\\n', ' ')
    cleaned = re.sub('\s+', ' ', cleaned)

    # Remove start and end white spaces
    cleaned = cleaned.strip()
    if cleaned != '':
        return cleaned


def init_preprocessing(data):
    # Remove special characters, outliers, duplicates, null values, html tags
    data = data.dropna()
    data['CommentText'] = data['CommentText'].apply(remove_special_characters)
    data = remove_outliers(data)
    # labels are already encoded(assume SimilarityScore is integer)
    return data


if __name__ == '__main__':
    raw_data = read_raw_data()
    init_variables(raw_data)
    preprocessed_data = init_preprocessing(raw_data)
    for step in list(range(9)):
        start_processing(step, preprocessed_data)