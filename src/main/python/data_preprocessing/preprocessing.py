import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.toktok import ToktokTokenizer

MAIN_CONFIG_DIR = '../../config'
PROCESSED_DATA_DIR = '../../resources/processed_data'

# TODO: should we use (1, 2), (1,3) and (2,3) combinations
# analyzer needed for multiple columns
cv_unigram = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False)
cv_bigram = CountVectorizer(ngram_range=(2, 2), lowercase=False)
cv_trigram = CountVectorizer(ngram_range=(3, 3), lowercase=False)
tokenizer = ToktokTokenizer()


def remove_stop_words_and_tokenize_word(text, stopwords):
    words = tokenizer.tokenize(text)
    return "".join([word + " " for word in words if word not in stopwords])


def read_stop_words():
    stopwords = []
    with open(f'{MAIN_CONFIG_DIR}/stopwords.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            # remove linebreak
            line_cleaned = line[:-1]
            stopwords.append(line_cleaned)
    return stopwords


def create_bag_of_words(data_set):
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
    np.savetxt(f'{file_path}', data_frame.to_numpy(), fmt="%d")


def prepare_files_for_stemming(data_frame, input_file_name):
    write_data_frame_to_file(data_frame, MAIN_CONFIG_DIR, input_file_name)


def do_file_stemming(data, input_file_name, output_file_name):
    prepare_files_for_stemming(data, input_file_name)
    execute_stemming_command(input_file_name, output_file_name)
    return get_stemming_result(output_file_name)


def stemming_and_remove_stopwords(data_set):
    stopwords = read_stop_words()
    data_set['CommentText'] = data_set['CommentText'].apply(lambda text: remove_stop_words_and_tokenize_word(text, stopwords))
    return create_bag_of_words(do_file_stemming(data_set, 'input-stemming.csv', 'output-stemming.csv'))


def bigrams(data_set):
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_bigram.fit(data_set["Merged Text"])
    print(cv_bigram.get_feature_names())
    bow = pd.DataFrame(cv_bigram.fit_transform(data_set["Merged Text"]).todense())
    return bow


def trigrams(data_set):
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
    4: trigrams
}


def read_raw_data():
    comments = pd.read_csv("../../resources/raw_data/comments.csv", names=['ProgrammingLanguageName', 'QueryID',
                                                                         'PairID', 'QueryText', 'CommentText',
                                                                         'SimilarityScore'])
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
        write_bow_data_frame_to_file(processed_data, f'{PROCESSED_DATA_DIR}/{processing_technique.__name__}.txt')


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
    data['QueryText'] = data['QueryText'].apply(remove_special_characters)
    data = remove_outliers(data)
    # labels are already encoded(assume SimilarityScore is integer)
    return data


if __name__ == '__main__':
    for step in list(range(9)):
        raw_data = read_raw_data()
        preprocessed_data = init_preprocessing(raw_data)
        start_processing(step, preprocessed_data)