def split_string_with_underscore(value):
    output = None
    if((value is None) == False):
        single_words_in_string_value = value.split()
        output = "_".join(single_words_in_string_value)
    return output

def split_string_with_plus(value):
    output = None
    if((value is None) == False):
        single_words_in_string_value = value.split()
        output = "+".join(single_words_in_string_value)
    return output