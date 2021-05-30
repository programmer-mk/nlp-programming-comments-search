# imports
from .classifiers import SVC


def apply_all_classifiers(data):
    SVC.support_vector_classifier(data)

    # TODO: do next two classifiers @djojdanic @bselic
    #multinomialNB.multinomial_nb(data)
    #logistic_regression.logistic_regression(data)


def load_preprocessed_data(file):
    pass


def classifying():
    print("----------  No preprocessing(BOW) ----------")
    apply_all_classifiers("without_preprocessing.txt")

    print("----------  Lower casing ----------")
    apply_all_classifiers("lowercasing.txt")

    print("----------  Term Frequency ----------")
    apply_all_classifiers("tf.txt")

    #double check this
    #print("----------  Inverse Document Frequency ----------")
    #apply_all_classifiers("idf.txt")

    print("----------  Term Frequency–Inverse Document Frequency ----------")
    apply_all_classifiers("tf_idf.txt")

    print("----------  Stemming and stopwords ----------")
    apply_all_classifiers("stemming_and_remove_stopwords.txt")

    print("----------  Frequency word filtering ----------")
    apply_all_classifiers("frequency_filtering.txt")

    print("----------  Bigram preprocessing ----------")
    apply_all_classifiers("bigrams.txt")

    print("----------  Trigram preprocessing ----------")
    apply_all_classifiers("trigrams.txt")

    print("----------  Binary Bag of Words ----------")
    apply_all_classifiers("binary_bow.txt")


if __name__ == "__main__":
    option = int(input("Choose option? \n"
                       "0 - classifying \n"
                       "1 - calculate comment annotation similarity \n"))

    # TODO: add comment annotation similarity @djojdanic @bselic
    if option == 0:
        classifying()
    else:
        print('percentage annotation calculation!')
        #percentage_calc.percentage_calculator()