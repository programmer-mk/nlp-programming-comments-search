from numpy.lib.utils import source
import pandas as pd
import numpy as np
import csv as csv
import validators as validator
import re

def remove_special_characters(text):
    # removes special characters with ' '
    cleaned = repr(text).strip("'")

    cleaned = re.sub('[^\\n\\t..,0-9a-zA-Z\u0080-\uFFFF]', ' ', text)
    cleaned = re.sub('_', ' ', cleaned)

    # Change any white space and new line to one space
    cleaned = cleaned.replace('\n', '\\n')
    cleaned = cleaned.replace('\t', '\\t')
    cleaned = re.sub(' +', ' ', cleaned)

    # # Remove start and end white spaces
    cleaned = cleaned.strip()
    if cleaned != '':
        return cleaned

def find_nth_substring_in_string(string, substring, n):
    try:
        parts = string.split(substring, n + 1)
        if len(parts) <= n + 1:
            return -1
        output = len(string) - len(parts[-1]) - len(substring)
    except Exception as e:
        print('Glupa funkcija')    
    return output

C_SHARP = 'C#'
COMMENT_TEXT = 'CommentText'
PAIR_ID = 'PairID'
REPO_DESCRIPTION = 'RepoDescription'
SOURCE_DESCRIPTION = 'SourceDescription'

COLUMN_MAPPING = {
    COMMENT_TEXT : 1,
    PAIR_ID: 2,
    REPO_DESCRIPTION : 3,
    SOURCE_DESCRIPTION : 4
}

excel_file_path = 'resources/programming_comments_annotation.xlsx'
excel_data = pd.read_excel(excel_file_path, engine = 'openpyxl')
excel_data = excel_data.drop(columns=['QueryText','SimilarityScore','Annotated_By'])
excel_data.insert(0, 'ProgrammingLanguageName', 'C#')

file_to_write_path = 'resources/pregled_svih_parova.txt'

with open(file_to_write_path, "w", encoding='utf8') as file_to_write:

    excel_header_write = 'ProgrammingLanguageName   RepoDescription SourceDescription   PairID  CommentText\n'

    file_to_write.write(excel_header_write)

    for excel_row in excel_data.values:

        try:
            source_description = excel_row[COLUMN_MAPPING[SOURCE_DESCRIPTION]]

            is_source_description_url = 'http' in source_description

            repo_description = excel_row[COLUMN_MAPPING[REPO_DESCRIPTION]]

            if is_source_description_url and source_description != 'nan' and len(source_description.strip()) > 0:
                index = find_nth_substring_in_string(source_description, '/', 4)
                repo_description = source_description[0:index]

            file_name = excel_row[COLUMN_MAPPING[PAIR_ID]]
            pair_id = file_name[0:len(file_name) - 4]

            comment_text = remove_special_characters(excel_row[COLUMN_MAPPING[COMMENT_TEXT]])

            excel_row_write = f'{C_SHARP} {repo_description}  {source_description} {pair_id}    {comment_text}\n'

            file_to_write.write(excel_row_write)

        except Exception as e:
            print(f"Oops! Error occurred in {excel_row}. {e}")
            print("Next entry.")
            print()
