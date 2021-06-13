from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import sys


RESOURCES_DIR = '../../resources'
operating_system = sys.platform

if operating_system == 'win32':
    RESOURCES_DIR = 'src\main/resources'


def read_raw_data():
    comments = pd.read_csv("../../../resources/raw_data/comments.csv", names=['ProgrammingLanguageName', 'QueryID',
                                                                           'PairID', 'QueryText', 'CommentText',
                                                                           'SimilarityScore'])
    return comments


def found_index_in_data(data, query_id, comment_text):
    return np.where((data['QueryID'] == query_id) & (data['CommentText'] == comment_text))[0][0]


def load_all_data():
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns, sep='\t')
    # drop header
    data_columns.drop(index=data_columns.index[0], axis=0, inplace=True)
    print('end loading query data..')
    return data_columns


def build_model(data_set_comments_bow, data_set_queries_bow):
    all_data = load_all_data()
    query_similarity_list = {}
    filtered_data = all_data[all_data['SimilarityScore'] != '0']

    for idx, row in filtered_data.iterrows():
        index = found_index_in_data(all_data,row[1], row[4])
        query = data_set_queries_bow.iloc[index].to_frame().values.reshape(1,-1)
        cos_similarity_list = list(map(lambda comment: cosine_similarity(query, comment.reshape(1,-1)), data_set_comments_bow.values))
        query_similarity_list[row[1]] = cos_similarity_list
    return query_similarity_list


def build_random_data_set(target_index, target_query_id,  data_set, test_data_size):
    test_data_set = data_set.drop([data_set.index[target_index]])
    cleaned_test_data = test_data_set[(test_data_set['SimilarityScore'] == 0) & (test_data_set['QueryID'] == target_query_id)]
    if test_data_size > cleaned_test_data.shape[0]:
        return cleaned_test_data.sample(cleaned_test_data.shape[0])
    else:
        return cleaned_test_data.sample(test_data_size)




def evaluate_model(data, model):
    non_zero_sim_data = data[data['SimilarityScore'] != 0]
    scores = 0.0
    for _, row in non_zero_sim_data.iterrows():
        target_index = found_index_in_data(data,row[1], row[4])
        cos_similarity_list = model.get(row[3])
        test_data_set = build_random_data_set(target_index, row[1], data, 99)
        # these indexes are for random test data in original data set
        test_data_indexes = list(map(lambda test_row: found_index_in_data(data, test_row[1]['QueryID'], test_row[1]['CommentText']), test_data_set.iterrows()))
        test_data_similarity_vals = sorted([cos_similarity_list[index][0][0] for index in test_data_indexes], reverse=True)
        index_place = np.searchsorted(test_data_similarity_vals, cos_similarity_list[target_index][0][0])
        scores += 1 / (index_place + 1)

    return scores / non_zero_sim_data.shape[0]



def start_ranking(data_set_comments, data_set_queries):
    model = build_model(data_set_comments, data_set_queries)
    print('Go to sleep please')


if __name__ == '__main__':
    pass
    #dataset = load_query_data()
    #model = build_model(data_set[['QueryText', 'CommentText']])
    #print(dataset)
    #data_set = read_raw_data()

    #mrr = evaluate_model(data_set, model)
    #print(f'mean reciprocal rank of model is : {mrr}')