import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

cv_unigram = CountVectorizer(ngram_range=(1, 1))
cv_bigram = CountVectorizer(ngram_range=(2, 2))
cv_trigram = CountVectorizer(ngram_range=(3, 3))


def without_preprocessing(data_set):
    pass


def lowercasing(data_set):
    pass


processing_steps = {
    1: without_preprocessing,
    2: lowercasing
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
    pass


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
    #labels are already encoded(assume SimilarityScore is integer)
    return data


if __name__ == '__main__':
    raw_data = read_raw_data()
    init_variables(raw_data)
    preprocessed_data = init_preprocessing(raw_data)
    for step in list(range(9)):
        start_processing(step, preprocessed_data)