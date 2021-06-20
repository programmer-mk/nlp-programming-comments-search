import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import operator
from wordcloud import WordCloud

operating_system = sys.platform
resources_directory = '../resources'
if(operating_system == 'win32'):
    resources_directory = 'src\main/resources'
    config_directory = 'src\main/config'


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
    vectorizer = CountVectorizer()
    vectorizer.fit(data['CommentText'])

    print("Vocabulary: ", vectorizer.vocabulary_)

    # look just for distinct comments
    for key in vectorizer.vocabulary_:
        value = vectorizer.vocabulary_[key]
        vectorizer.vocabulary_[key] = value / 99

    result_dict = {
        '30-35': 0,
        '25-30': 0,
        '20-25': 0,
        '15-20': 0,
        '10-15': 0,
        '5-10': 0,
        '2-5': 0,
        '1-2': 0
    }

    for key in vectorizer.vocabulary_:
        value = vectorizer.vocabulary_[key]
        if 30 <= value <= 35:
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
        elif 2 <= value < 5:
            result_dict['2-5'] = result_dict.get('2-5', 0) + 1
        else:
            result_dict['1-2'] = result_dict.get('1-2', 0) + 1


    plt.bar(*zip(*result_dict.items()))
    plt.xticks(rotation='vertical')
    plt.show()


def comment_wordcloud(data):
    vectorizer = CountVectorizer()
    vectorizer.fit(data['CommentText'])
    print("Vocabulary: ", vectorizer.vocabulary_)

    wordcloud = WordCloud(width=1000,height=1000).generate_from_frequencies(vectorizer.vocabulary_)
    plt.figure(figsize=(9,6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def comment_word_frequency(data):
    vectorizer = CountVectorizer()
    vectorizer.fit(data['CommentText'])

    print("Vocabulary: ", vectorizer.vocabulary_)
    # look just for distinct comments
    for key in vectorizer.vocabulary_:
        value = vectorizer.vocabulary_[key]
        vectorizer.vocabulary_[key] = value / 99

    top_20_words_vocabulary = dict(sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1), reverse=True)[:20])
    print("Top N Vocabulary: ", top_20_words_vocabulary)
    plt.bar(*zip(*top_20_words_vocabulary.items()))
    plt.xticks(rotation='vertical')
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
    ss = data[data['SimilarityScore'] != '0']['QueryText'].value_counts()
    print(ss)
    if top:
        chart = data[data['SimilarityScore'] != '0']['QueryText'].value_counts().nlargest(10).plot.bar()
    else:
        chart = data[data['SimilarityScore'] != '0']['QueryText'].value_counts().nsmallest(10).plot.bar()

    plt.xticks(size=8, rotation='70')
    fig = chart.get_figure()
    fig.set_figwidth(15)
    fig.set_figheight(15)
    if top:
        fig.savefig('query_similarity_score_distribution_largest.pdf')
    else:
        fig.savefig('query_similarity_score_distribution_smallest.pdf')


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
                       "10 -  most freqeunt words in ranges \n"

        option = int(input(menu_message))
        if option >= 1 or option <= 9:
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