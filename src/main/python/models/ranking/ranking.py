from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import sys


RESOURCES_DIR = '../../../resources'
operating_system = sys.platform

if operating_system == 'win32':
    RESOURCES_DIR = 'src\main/resources'


def read_raw_data():
    comments = pd.read_csv("../../../resources/raw_data/comments.csv", names=['ProgrammingLanguageName', 'QueryID',
                                                                           'PairID', 'QueryText', 'CommentText',
                                                                           'SimilarityScore'])
    return comments


def load_query_data():
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns, sep='\t')[['QueryID', 'QueryText']]
    data_columns.drop_duplicates(keep='last', inplace=True)

    dddata = data_columns.value_counts()
    dddata.to_csv(f'tttest', sep = '\t', index = True)
    # DO NOT RESET INDEX!!!
    #data_columns.reset_index(drop=True, inplace=True)
    print('end loading query data..')
    return data_columns


def build_model(data_set_comments, data_set_queries):
    query_metadata = load_query_data()
    query_similarity_list = {}
    count_vectorizer = CountVectorizer(analyzer='word', lowercase=False)
    bow_comments = count_vectorizer.fit_transform((data_set['CommentText']))
    for query in query_metadata:
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
        test_data_set = build_random_data_set(target_index, row[1], data, 99)
        # these indexes are for random test data in original data set
        test_data_indexes = list(map(lambda test_row: found_index_in_data(data, test_row[1]['QueryID'], test_row[1]['CommentText']), test_data_set.iterrows()))
        test_data_similarity_vals = sorted([cos_similarity_list[index][0][0] for index in test_data_indexes], reverse=True)
        index_place = np.searchsorted(test_data_similarity_vals, cos_similarity_list[target_index][0][0])
        scores += 1 / (index_place + 1)

    return scores / non_zero_sim_data.shape[0]


if __name__ == '__main__':
    dataset = load_query_data()
    print(dataset)
    #data_set = read_raw_data()
    #model = build_model(data_set[['QueryText', 'CommentText']])
    #mrr = evaluate_model(data_set, model)
    #print(f'mean reciprocal rank of model is : {mrr}')