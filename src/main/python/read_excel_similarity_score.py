from numpy.lib.utils import source
import pandas as pd
import numpy as np
import validators as validator
import re

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

data_frame = pd.read_excel('resources/programming_comments_annotation.xlsx')

data_frame.drop(inplace = True, columns=['QueryText','SimilarityScore','Annotated_By'])
data_frame.insert(0, 'ProgrammingLanguageName', 'C#')
data_frame['CommentText'] = data_frame.apply(lambda row : remove_special_characters(row['CommentText']), axis=1)
data_frame.to_csv('resources/pregled_svih_parova_novi.txt', sep = '\t', index = False)

dict_query_line = {}

with open ('../config/queries_serbian.txt', 'r') as read_file_queries:
    queries = read_file_queries.readlines()
    cursor = 1
    for query in queries:
        dict_query_line[query] = cursor
        cursor += 1

for row in data_frame.index:

    queries = data_frame['QueryText'][row]
    similarity_scores = data_frame['SimilarityScore'][row]

    if '#' in queries:
        query_list = queries.split('#')
        similarity_score_list = similarity_scores.split('#')
    elif ',' in queries:
        query_list = queries.split(',')
        similarity_score_list = similarity_scores.split(',')
    else:
        query_list = queries
        similarity_score_list = similarity_scores

    for query in query_list:
        daf










