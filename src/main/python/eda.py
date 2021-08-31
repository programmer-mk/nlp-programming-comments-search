import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import operator
from wordcloud import WordCloud
import numpy as np
from collections import Counter

operating_system = sys.platform
resources_directory = '../resources'
if(operating_system == 'win32'):
    resources_directory = 'src\main/resources'
    config_directory = 'src\main/config'

def query_frequency_range(data):
    word_dict = data[data['SimilarityScore'] != '0']['QueryText'].value_counts().to_dict()

    result_dict = {
        '40-45': 0,
        '35-40': 0,
        '30-35': 0,
        '25-30': 0,
        '20-25': 0,
        '15-20': 0,
        '10-15': 0,
        '5-10': 0,
        '0-5': 0
    }

    for key in word_dict.keys():
        value = word_dict[key]
        if 40 <= value <= 45:
            result_dict['40-45'] = result_dict.get('40-45', 0) + 1
        elif 35 <= value < 40:
            result_dict['35-40'] = result_dict.get('35-40', 0) + 1
        elif 30 <= value < 35:
            result_dict['30-35'] = result_dict.get('30-35', 0) + 1
        elif 25 <= value < 30:
            result_dict['25-30'] = result_dict.get('25-30', 0) + 1
        elif 20 <= value < 25:
            result_dict['20-25'] = result_dict.get('20-25', 0) + 1
        elif 15 <= value < 20:
            result_dict['15-20'] = result_dict.get('15-20', 0) + 1
        elif 10 <= value < 15:
            result_dict['10-15'] = result_dict.get('10-15', 0) + 1
        elif 5 <= value < 10:
            result_dict['5-10'] = result_dict.get('5-10', 0) + 1
        else:
            result_dict['0-5'] = result_dict.get('0-5', 0) + 1


    plt.bar(*zip(*result_dict.items()))
    plt.xticks(rotation='vertical')
    plt.show()


'''
Ranges:

    1) 1-2
    2) 2-5
    3) 5-10
    4) 10-15
    5) 15-20
    6) 20-25
    7) 25-30
    8) 30-35
'''
def comment_word_frequency_range(data):
    result = Counter(" ".join(data['CommentText'].values.tolist()).split(" "))
    word_dict = dict(list(result.items()))
    # look just for distinct comments
    for key in word_dict.keys():
        value = word_dict[key]
        word_dict[key] = value / 99

    result_dict = {
        '300-350': 0,
        '250-300': 0,
        '200-250': 0,
        '150-200': 0,
        '100-150': 0,
        '50-100': 0,
        '25-50': 0,
        '20-25': 0,
        '15-20': 0,
        '10-15': 0,
        '5-10': 0,
        '0-5': 0
    }

    for key in word_dict.keys():
        value = word_dict[key]
        if 300 <= value <= 350:
            result_dict['300-350'] = result_dict.get('300-350', 0) + 1
        elif 250 <= value < 300:
            result_dict['250-300'] = result_dict.get('250-300', 0) + 1
        elif 200 <= value < 250:
            result_dict['200-250'] = result_dict.get('200-250', 0) + 1
        elif 150 <= value < 200:
            result_dict['150-200'] = result_dict.get('150-200', 0) + 1
        elif 100 <= value < 150:
            result_dict['100-150'] = result_dict.get('100-150', 0) + 1
        elif 50 <= value < 100:
            result_dict['50-100'] = result_dict.get('50-100', 0) + 1
        elif 25 <= value < 50:
            result_dict['25-50'] = result_dict.get('25-50', 0) + 1
        elif 20 <= value < 25:
            result_dict['20-25'] = result_dict.get('20-25', 0) + 1
        elif 15 <= value < 20:
            result_dict['15-20'] = result_dict.get('15-20', 0) + 1
        elif 10 <= value < 15:
            result_dict['10-15'] = result_dict.get('10-15', 0) + 1
        elif 5 <= value < 10:
            result_dict['5-10'] = result_dict.get('5-10', 0) + 1
        else:
            result_dict['0-5'] = result_dict.get('0-5', 0) + 1


    plt.bar(*zip(*result_dict.items()))
    plt.xticks(rotation='vertical')
    plt.show()


def count_avgerage_lengths_per_similarity_score(data):
    data['comment_len'] = data['CommentText'].astype(str).apply(len)
    data['word_count'] = data['CommentText'].apply(lambda x: len(str(x).split()))

    zeroes_avg_comm_len = data[data['SimilarityScore'] == '0']['comment_len'].mean()
    ones_avg_comm_len = data[data['SimilarityScore'] == '1']['comment_len'].mean()
    twoes_avg_comm_len = data[data['SimilarityScore'] == '2']['comment_len'].mean()
    threes_avg_comm_len = data[data['SimilarityScore'] == '3']['comment_len'].mean()

    zeroes_avg_word_count = data[data['SimilarityScore'] == '0']['word_count'].mean()
    ones_avg_word_count = data[data['SimilarityScore'] == '1']['word_count'].mean()
    twoes_avg_word_count = data[data['SimilarityScore'] == '2']['word_count'].mean()
    threes_avg_word_count = data[data['SimilarityScore'] == '3']['word_count'].mean()

    labels = ['0', '1', '2', '3']
    word_count_means = [zeroes_avg_word_count, ones_avg_word_count, twoes_avg_word_count, threes_avg_word_count]
    comment_len_means = [zeroes_avg_comm_len, ones_avg_comm_len, twoes_avg_comm_len, threes_avg_comm_len]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, word_count_means, width, label='Word Count')
    rects2 = ax.bar(x + width/2, comment_len_means, width, label='Sentence Lenght')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_xlabel('SimilarityScore')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()


def comment_wordcloud(data):
    result = Counter(" ".join(data['CommentText'].values.tolist()).split(" "))
    word_dict = dict(list(result.items()))
    # look just for distinct comments
    for key in word_dict.keys():
        value = word_dict[key]
        word_dict[key] = value / 99

    wordcloud = WordCloud(width=1000,height=1000).generate_from_frequencies(word_dict)
    plt.figure(figsize=(9,6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def comment_word_frequency(data):
    result = Counter(" ".join(data['CommentText'].values.tolist()).split(" "))
    word_dict = dict(list(result.items()))
    # look just for distinct comments
    for key in word_dict.keys():
        value = word_dict[key]
        word_dict[key] = value / 99

    top_20_words_vocabulary = dict(sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)[:20])
    print("Top N Vocabulary: ", top_20_words_vocabulary)
    plt.bar(*zip(*top_20_words_vocabulary.items()))
    plt.xticks(weight='bold')
    plt.show()


def load_eda_data():
    columns = ['ProgrammingLanguage', 'QueryId','PairID', 'QueryText', 'CommentText','SimilarityScore']
    data = pd.read_csv(f"{resources_directory}/output_similarity_score.csv", sep="\t", names=columns)
    data.drop(index=data.index[0], axis=0, inplace=True)

    # add annotator information
    data['Annotator'] = ''
    for ind in data.index:
        if 'mkovacevic' in data['PairID'][ind]:
            data['Annotator'][ind] = 'mkovacevic'
        elif 'bselic' in data['PairID'][ind]:
            data['Annotator'][ind] = 'bselic'
        else:
            data['Annotator'][ind] = 'djojdanic'

    return data


def collected_data_per_annotator(data):
    chart = data['Annotator'].value_counts().plot(kind='pie', autopct='%1.2f%%')
    fig = chart.get_figure()
    fig.savefig(f'collected_data_per_annotator.pdf')


def similarity_score_per_query(data, top=True):
    if top:
        top10 = data[data['SimilarityScore'] != '0']['QueryText'].value_counts().nlargest(10)
    else:
        top10 = data[data['SimilarityScore'] != '0']['QueryText'].value_counts().nsmallest(10)

    values = top10.values.tolist()
    labels = ['\n'.join(l.split(" ")) for l in top10.index.tolist()]
    plt.figure()
    plt.bar(list(range(0, 10)), values)
    plt.xticks(list(range(0, 10)), labels)
    plt.show()


def similarity_score_value_distribution_per_annotator(data, value):
    chart = data[data['SimilarityScore'] == str(value)]['Annotator'].value_counts().plot(kind='pie', autopct='%1.2f%%')
    fig = chart.get_figure()
    fig.savefig(f'{value}_distribution.pdf')


def similarity_score_distribution(df):
    # display histogram of non zero values for SimilarityScore
    chart = df[df['SimilarityScore'] != '0']['SimilarityScore'].hist()
    fig = chart.get_figure()
    fig.savefig(f'similarity_score_distribution.pdf')


visualizations = {
    1: similarity_score_distribution,
    2: collected_data_per_annotator,
    3: similarity_score_value_distribution_per_annotator,
    4: similarity_score_value_distribution_per_annotator,
    5: similarity_score_value_distribution_per_annotator,
    6: similarity_score_per_query,
    7: similarity_score_per_query,
    8: comment_word_frequency,
    9: comment_wordcloud,
    10: comment_word_frequency_range,
    11: count_avgerage_lengths_per_similarity_score,
    12: query_frequency_range
}


if __name__ == '__main__':
    correct_input = False

    while not correct_input:
        menu_message = "Choose option? \n" \
                       "1 - similarity score distribution \n" \
                       "2 - collected data per annotator \n" \
                       "3 - similarity score value 1 distribution per annotator \n" \
                       "4 - similarity score value 2 distribution per annotator \n" \
                       "5 - similarity score value 3 distribution per annotator \n" \
                       "6 -  similarity score per query top 10 largest \n" \
                       "7 - similarity score per query top 10 smallest \n" \
                       "8 - most frequent words \n" \
                       "9 - most frequent words - wordcloud \n" \
                       "10 -  most freqeunt words in ranges \n" \
                       "11 - average comment lengths per similarity score \n" \
                       "12 - most freqeunt queries in ranges \n"

        option = int(input(menu_message))
        if option >= 1 or option <= 12:
            print('correct input. Procced with computation and visualization')
            data = load_eda_data()
            correct_input = True

            visualization = visualizations.get(option)
            if 3 <= option <= 5:
                visualization(data, option-2)
            elif option == 7:
                visualization(data, False)
            else:
                visualization(data)

        else:
            print('Wrong input.')