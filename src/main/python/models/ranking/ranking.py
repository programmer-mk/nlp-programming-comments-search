from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np


def read_raw_data():
    comments = pd.read_csv("../../../resources/raw_data/comments.csv", names=['ProgrammingLanguageName', 'QueryID',
                                                                           'PairID', 'QueryText', 'CommentText',
                                                                           'SimilarityScore'])
    return comments


def build_model(data_set):
    query_similarity_list = {}
    count_vectorizer = CountVectorizer(analyzer='word', lowercase=False)
    bow_comments = count_vectorizer.fit_transform((data_set['CommentText']))
    queries = data_set['QueryText'].drop_duplicates(keep='last')
    for query in queries:
        query_frame = pd.DataFrame([query], columns=['QueryText'])
        query_vectorized = count_vectorizer.transform(query_frame['QueryText'])
        cos_similarity_list = list(map(lambda comment: cosine_similarity(query_vectorized, comment), bow_comments))
        print(cos_similarity_list)
        query_similarity_list[query] = cos_similarity_list
    return query_similarity_list


def build_random_data_set(target_index, target_query_id,  data_set, test_data_size):
    test_data_set = data_set.drop([data_set.index[target_index]])
    cleaned_test_data = test_data_set[(test_data_set['SimilarityScore'] == 0) & (test_data_set['QueryID'] == target_query_id)]
    if test_data_size > cleaned_test_data.shape[0]:
        return cleaned_test_data.sample(cleaned_test_data.shape[0])
    else:
        return cleaned_test_data.sample(test_data_size)


def found_index_in_data(data, query_id, comment_text):
    return np.where((data['QueryID'] == query_id) & (data['CommentText'] == comment_text))[0][0]


def evaluate_model(data, model):
    non_zero_sim_data = data[data['SimilarityScore'] != 0]
    scores = 0.0
    for _, row in non_zero_sim_data.iterrows():
        target_index = found_index_in_data(data,row[1], row[4])
        cos_similarity_list = model.get(row[3])
        test_data_set = build_random_data_set(target_index, row[1], data, 2)
        # these indexes are for random test data in original data set
        test_data_indexes = list(map(lambda test_row: found_index_in_data(data, test_row[1]['QueryID'], test_row[1]['CommentText']), test_data_set.iterrows()))
        test_data_similarity_vals = list(map(lambda index: cos_similarity_list[index][0][0], test_data_indexes)).sort(reverse=True)
        index_place = np.searchsorted(test_data_similarity_vals, cos_similarity_list[target_index][0][0])
        scores = 1 / (index_place + 1)

    return scores/ non_zero_sim_data.shape[0]


if __name__ == '__main__':
    data_set = read_raw_data()
    model = build_model(data_set[['QueryText', 'CommentText']])
    mrr = evaluate_model(data_set, model)
    print(f'mean reciprocal rank of model is : {mrr}')