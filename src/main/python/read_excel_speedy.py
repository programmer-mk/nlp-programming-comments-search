from numpy.lib.utils import source
import pandas as pd
import numpy as np
import re
import sys

all_queries = []

def remove_special_characters(text):
    # removes special characters with ' '
    cleaned = repr(text).strip("'")

    cleaned = re.sub('[^\\n\\t..,0-9a-zA-Z\u0080-\uFFFF]', ' ', str(text))
    cleaned = re.sub('_', ' ', str(cleaned))

    # Change any white space and new line to one space
    cleaned = cleaned.replace('\n', '\\n')
    cleaned = cleaned.replace('\t', '\\t')
    cleaned = re.sub(' +', ' ', str(cleaned))

    # # Remove start and end white spaces
    cleaned = cleaned.strip()
    if cleaned != '':
        return cleaned


def trim_pair_id(pair_id):
    return pair_id[0:len(pair_id)-4]

operating_system = sys.platform

resources_directory = '../resources'
config_directory = '../config'

if(operating_system == 'win32'):
    resources_directory = 'src\main/resources'
    config_directory = 'src\main/config'

data_frame_global = pd.read_excel(f'{resources_directory}/programming_comments_annotation.xlsx')
data_frame_global.dropna(how='any', inplace=True)
data_frame_global['CommentText'] = data_frame_global.apply(lambda row : remove_special_characters(row['CommentText']), axis=1)
data_frame_global['PairID'] = data_frame_global.apply(lambda row : trim_pair_id(row['PairID']), axis=1)

data_frame = data_frame_global.drop(columns=['QueryText','SimilarityScore','Annotated_By'])
data_frame.insert(0, 'ProgrammingLanguageName', 'C#')
data_frame['CommentText'] = data_frame.apply(lambda row : remove_special_characters(row['CommentText']), axis=1)
data_frame.columns = ['ProgrammingLanguage', 'RepoDescription', 'SourceDescription', 'PairID', 'CommentText']
data_frame.to_csv(f'{resources_directory}/pregled_svih_parova_novi.txt', sep = '\t', index = False)

data_frame_similarity_score = pd.DataFrame(columns=['ProgrammingLanguage', 'QueryId', 'PairID', 'QueryText', 'CommentText','SimilarityScore'])
dict_query_line = {}

with open (f'{config_directory}/queries_serbian.txt', 'r') as read_file_queries:
    for index, query in enumerate(read_file_queries.readlines()):
        query = query.replace('\n', '')
        all_queries.append(query)
        dict_query_line[query] = index

for row in data_frame_global.index:
    queries = str(data_frame_global['QueryText'][row])
    similarity_scores = str(data_frame_global['SimilarityScore'][row])
    
    if '#' in queries:
        query_list = queries.split('#')
    elif ',' in queries:
        query_list = queries.split(',')
    else:
        query_list = [queries]

    if '#' in similarity_scores:
        similarity_score_list = similarity_scores.split('#')
    elif ',' in similarity_scores:
        similarity_score_list = similarity_scores.split(',')
    else:
        similarity_score_list = [similarity_scores]

    similarity_score_list = similarity_score_list[0:len(query_list)]
    query_list = query_list[0:len(similarity_score_list)]

    if(type(similarity_score_list) == str):
        similarity_score_list = [similarity_score_list]

    if(type(query_list) == str):
        query_list = [query_list]

    rest_queries = list(filter(lambda query: str(query) not in query_list, all_queries))

    for index, query in enumerate(query_list):
        query_id = dict_query_line.get(query)
        if query_id == 0 or query_id == None:
            print(query)
        else:
            print(f'Processing row: {row}')
            dict_similarity_score = {
                'ProgrammingLanguage' : 'C#',
                'QueryId' : query_id,
                'PairID' :  data_frame_global['PairID'][row],
                'QueryText' :  query,
                'CommentText' : data_frame_global['CommentText'][row],
                'SimilarityScore': similarity_score_list[index]
            }
            data_frame_similarity_score = data_frame_similarity_score.append(pd.DataFrame([dict_similarity_score]), ignore_index=True)

    for index, rest_query in enumerate(rest_queries):
        rest_query_id = dict_query_line.get(rest_query)
        if rest_query_id == None:
            print(rest_query)
        else:    
            dict_similarity_score = {
                'ProgrammingLanguage' : 'C#',
                'QueryId' : rest_query_id + 1,
                'PairID' :  data_frame_global['PairID'][row],
                'QueryText' :  rest_query,
                'CommentText' : data_frame_global['CommentText'][row],
                'SimilarityScore': 0
            }
            data_frame_similarity_score = data_frame_similarity_score.append(pd.DataFrame([dict_similarity_score]), ignore_index=True)

data_frame_similarity_score.to_csv(f'{resources_directory}/pregled_svih_similarity_score.txt', sep = '\t', index = False)
data_frame_similarity_score.to_csv(f'{resources_directory}/pregled_svih_similarity_score.csv', sep = '\t', index = False)









