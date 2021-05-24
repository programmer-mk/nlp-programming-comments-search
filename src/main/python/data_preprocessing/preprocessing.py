import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

cv_unigram = CountVectorizer(ngram_range=(1, 1))
cv_bigram = CountVectorizer(ngram_range=(2, 2))
cv_trigram = CountVectorizer(ngram_range=(3, 3))


def without_preprocessing():
    pass


def lowercasing():
    pass


processing_steps = {
    1: without_preprocessing(),
    2: lowercasing()
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


def create_bag_of_words(data_set):
    return cv_unigram.transform(data_set['Comment'])


def start_preprocessing(step, data_set):
    function = processing_steps.get(step)
    if function is None:
        print('Unrecognized processing type. Please specify number between 1 - x')
    else:
        function(data_set)


if __name__ == '__main__':
    init_data_set = read_raw_data()
    init_variables(init_data_set)
    for step in list(range(9)):
        start_preprocessing(step, init_data_set)