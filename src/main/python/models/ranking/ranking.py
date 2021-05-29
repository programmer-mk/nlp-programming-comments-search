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


def build_random_data_set(target_entry, data_set, test_data_size):
    test_data_set = data_set[(data_set['SimilarityScore']) == 0 & (data_set['QueryText'] == target_entry[3])].sample(test_data_size) # should be 99 or 74
    #test_data_set= pd.concat([filter_data_set, target_entry], ignore_index=True, sort=False)
    return test_data_set


def found_index_in_data(data, coulmn_name, query_id):
    return np.where(data[coulmn_name] == query_id)[0][0]


def evaluate_model(data, model):
    non_zero_sim_data = data[data['SimilarityScore'] != 0]
    for index, row in non_zero_sim_data.iterrows():
        target_index = found_index_in_data(data,'QueryID',row[1])
        cos_similarity_list = model.get(row[3])
        test_data_set = build_random_data_set(row, data, 2)
        # these indexes are for random test data in original data set
        test_data_indexes = list(map(lambda test_row: found_index_in_data(data,'QueryID', test_row[1]), test_data_set))
        print(test_data_set)


if __name__ == '__main__':
    data_set = read_raw_data()
    model = build_model(data_set[['QueryText', 'CommentText']])
    mrr = evaluate_model(data_set, model)
    print(f'mean reciprocal rank of model is : {mrr}')