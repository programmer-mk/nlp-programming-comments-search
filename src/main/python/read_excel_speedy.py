from numpy.lib.utils import source
import pandas as pd
import numpy as np
import validators as validator
import re

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

data_frame_global = pd.read_excel('resources/programming_comments_annotation.xlsx')

data_frame = data_frame_global.drop(columns=['QueryText','SimilarityScore','Annotated_By'])
data_frame.insert(0, 'ProgrammingLanguageName', 'C#')
data_frame['CommentText'] = data_frame.apply(lambda row : remove_special_characters(row['CommentText']), axis=1)
data_frame.to_csv('resources/pregled_svih_parova_novi.txt', sep = '\t', index = False)

data_frame_similarity_score = pd.DataFrame(columns=['QueryText','CommentText','PairID','RepoDescription', 'SourceDescription', 'SimilarityScore', 'Annotated_By'])

dict_query_line = {}

with open ('src\main/config/queries_serbian.txt', 'r') as read_file_queries:
    all_queries = read_file_queries.readlines()
    for index, query in enumerate(all_queries):
        query = query.replace('\n', '')
        dict_query_line[query] = index

for row in data_frame_global.index:
    queries = str(data_frame_global['QueryText'][row])
    rest_queries = filter(lambda query: str(query) not in all_queries, all_queries)
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
            data_frame_similarity_score.append(DataFrame=pd.DataFrame(dict_similarity_score), index = [index])

    for index, rest_query in enumerate(rest_queries):
        rest_query_id = dict_query_line.get(rest_query)
        if rest_query_id == None:
            print(rest_query)
        else:    
            dict_similarity_score = {
                'ProgrammingLanguage' : 'C#',
                'QueryId' : rest_query_id,
                'PairID' :  data_frame_global['PairID'][row],
                'QueryText' :  rest_query,
                'CommentText' : data_frame_global['CommentText'][row],
                'SimilarityScore': 0
            }
            data_frame_similarity_score.append(DataFrame=pd.DataFrame(dict_similarity_score), index = [index])    

data_frame_similarity_score.to_csv('resources/pregled_svih_similarity_score.txt', sep = '\t', index = False)









