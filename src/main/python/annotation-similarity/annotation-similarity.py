import pandas as pd
import sys

operating_system = sys.platform
resources_directory = '../../resources'
if(operating_system == 'win32'):
    resources_directory = 'src\main/resources'
    config_directory = 'src\main/config'

# diff  (unit - percentage)
similarity = {
    0: 100,
    1: 66,
    2: 33,
    3: 0,
}


annotators = {
    1: 'Bojan Selic',
    2: 'Milisav Kovacevic',
    3: 'Djordje Ojdanic'
}


def found_percentage_diff(scores1, scores2, row):
    similarity_total = 0
    if len(scores1) == len(scores2):
        for idx, score in enumerate(scores1):
            similarity_total += similarity.get(abs(scores1[idx] - scores2[idx]))
    else:
        print(f'row {row} is not annotated well.')

    return similarity_total / len(scores1)


def compute_similarity(data_frame_first_annotator, data_frame_second_annotator, data_frame_third_annotator):
    first_second_similariy = 0
    second_third_similariy = 0
    first_third_similariy = 0
    samples_size = data_frame_first_annotator.shape[0]

    for row in data_frame_first_annotator.index:
        first_bucket = data_frame_first_annotator['SimilarityScore'][row]
        second_bucket = data_frame_second_annotator['SimilarityScore'][row]
        third_bucket = data_frame_third_annotator['SimilarityScore'][row]

        if isinstance(first_bucket, int):
            first_bucket = str(first_bucket)

        if isinstance(second_bucket, int):
            second_bucket = str(second_bucket)

        if isinstance(third_bucket, int):
            third_bucket = str(third_bucket)

        data1 = list(map(int, first_bucket.split(',')))
        data2 = list(map(int, second_bucket.split(',')))
        data3 = list(map(int, third_bucket.split(',')))
        first_second_similariy += found_percentage_diff(data1, data2, row)
        second_third_similariy += found_percentage_diff(data2, data3, row)
        first_third_similariy += found_percentage_diff(data1, data3, row)

    return first_second_similariy / samples_size, second_third_similariy / samples_size, first_third_similariy / samples_size


def read_annotated_data():
    data_frame_bselic = pd.read_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_bselic.xlsx")
    data_frame_mkovacevic = pd.read_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_mkovacevic.xlsx")
    data_frame_djojdanic = pd.read_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_djojdanic.xlsx")
    return data_frame_bselic, data_frame_mkovacevic, data_frame_djojdanic


def extract_test_set(start_index, end_index):
    data_frame = pd.read_excel(f'{resources_directory}/programming_comments_annotation.xlsx')
    filtered_dataframe = data_frame[start_index:end_index]
    filtered_dataframe.to_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_bselic.xlsx", sheet_name='BSelic Annotation')
    filtered_dataframe.to_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_mkovacevic.xlsx", sheet_name='MKovacevic Annotation')
    filtered_dataframe.to_excel(f"{resources_directory}/annotation-similarity/shared_programming_comments_annotation_djojdanic.xlsx", sheet_name='Djojdanic Annotation')


if __name__ == '__main__':
    # enable this flag at beginning of process, to get 75  rows from original excel
    needExtract = False

    if needExtract:
        extract_test_set(575, 650)
    else:
        dataframe1, dataframe2, dataframe3 = read_annotated_data()
        similarity12, similarity23, similarity13 = compute_similarity(dataframe1, dataframe2, dataframe3)
        print(f'Annotation similarity between {annotators.get(1)} and {annotators.get(2)} is : {similarity12}')
        print(f'Annotation similarity between {annotators.get(2)} and {annotators.get(3)} is : {similarity23}')
        print(f'Annotation similarity between {annotators.get(1)} and {annotators.get(3)} is : {similarity13}')
