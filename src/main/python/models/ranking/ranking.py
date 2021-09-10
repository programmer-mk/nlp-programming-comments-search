from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import sys

all_data = None

RESOURCES_DIR = '../../resources'
operating_system = sys.platform

if operating_system == 'win32':
    RESOURCES_DIR = 'src\main/resources'


def found_index_in_data(data, query_id, comment_text):
    return data[(data['QueryID'] == query_id) & (data['CommentText'] == comment_text)].index.values[0]


def load_all_data():
    columns = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
    data_columns = pd.read_csv(f"{RESOURCES_DIR}/output_similarity_score.csv", names=columns, sep='\t')
    # drop header
    data_columns.drop(index=data_columns.index[0], axis=0, inplace=True)
    ss = data_columns['SimilarityScore'].value_counts()
    print('end loading query data..')
    return data_columns


def process_ranking_query(query, all_data, data_set_queries_bow, data_set_comments_bow):
    print(f'Ranking -- query in consideration: {query}')
    index = all_data[all_data['QueryText'] == query].index.values[0]
    print('1')
    query_bow = data_set_queries_bow.iloc[index].to_frame().values.reshape(1,-1)
    print('2')
    data_set_comments_bow = data_set_comments_bow.drop_duplicates(keep='first')
    print(f'process_ranking_query - after removing duplicates: {data_set_comments_bow.shape[0]}')
    cos_similarity_list = list(map(lambda comment: cosine_similarity(query_bow, comment.reshape(1,-1)), data_set_comments_bow.values))
    query_similarity_list  = {}
    query_similarity_list[query] = cos_similarity_list
    return query_similarity_list


def build_model(data_set_comments_bow, data_set_queries_bow):
    global all_data
    all_data = load_all_data()
    distinct_queries = all_data[['QueryText']].drop_duplicates(keep='first').values.tolist()
    print(distinct_queries)

    print('start training ranking model...')
    results = Parallel(n_jobs=-1)(delayed(process_ranking_query)(query[0], all_data, data_set_queries_bow, data_set_comments_bow) for query in distinct_queries)
    query_similarity_list = {}
    for dict in results:
        query_similarity_list.update(dict)
    return query_similarity_list


def build_random_data_set(target_index, target_query_id,  data_set, test_data_size):
    test_data_set = data_set.drop([data_set.index[target_index-1]])
    cleaned_test_data = test_data_set[(test_data_set['SimilarityScore'] == '0') & (test_data_set['QueryID'] == target_query_id)]
    cleaned_test_data = cleaned_test_data.sample(test_data_size)
    cleaned_test_data = cleaned_test_data.append(data_set.iloc[target_index-1])
    return cleaned_test_data


def evaluate_model(model):
    global all_data
    non_zero_sim_data = all_data[all_data['SimilarityScore'] != '0']
    scores = 0.0
    for _, row in non_zero_sim_data.iterrows():
        target_index = found_index_in_data(all_data,row[1], row[4])
        cos_similarity_list = model.get(row[3])
        test_data_set = build_random_data_set(target_index, row[1], all_data, 99)
        # these indexes are for random test data in original data set
        test_data_indexes = test_data_set.index.values.tolist()
        test_data_similarity_vals = sorted([cos_similarity_list[index-1][0][0] for index in test_data_indexes], reverse=True)

        index_place = 0
        for comp_value in test_data_similarity_vals:
            if comp_value > cos_similarity_list[target_index-1][0][0]:
                index_place += 1

        #  np.searchsorted(test_data_similarity_vals, cos_similarity_list[target_index-1][0][0])
        scores += 1 / (index_place + 1)

    return scores / non_zero_sim_data.shape[0]


def start_ranking(data_set_comments, data_set_queries):
    model = build_model(data_set_comments, data_set_queries)
    mrr = evaluate_model(model)
    print(f'Mean reciprocial rank is: {mrr}')
    print('Go to sleep please')

"""
    this main function is used just to validate cosinus similarity, functions are not called from here in original project flow
"""
if __name__ == '__main__':
    data_set = load_all_data()
    cv_unigram = CountVectorizer(ngram_range=(1, 1), analyzer='word', lowercase=False)
    data_set["Merged Text"] = data_set["CommentText"] + ' ' + data_set["QueryText"]
    cv_unigram.fit(data_set["Merged Text"])
    bow_comments = pd.DataFrame(cv_unigram.transform(data_set["CommentText"]).todense())
    bow_queries = pd.DataFrame(cv_unigram.transform(data_set["QueryText"]).todense())
    index = data_set[(data_set['QueryText'] == 'dohvati putanju izvrsnog fajla') & (data_set['CommentText'] == 'Upisuje sadrzaj liste u json fajl. Postojeci sadrzaj ce biti zamenjen novim.')].index.values[0]
    val = cosine_similarity(bow_comments.iloc[index].values.reshape(1,-1), bow_queries.iloc[index].values.reshape(1,-1))
    print(val)