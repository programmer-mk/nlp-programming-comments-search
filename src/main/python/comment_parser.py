import os
import pathlib
import re
import tokenize

MAIN_RESOURCE_DIR = '../resources'
JAVA_FILE_EXTENSION = '.java'
PYTHON_FILE_EXTENSION = '.py'

# TODO: add C# here
extension_mapping = {
    'java': JAVA_FILE_EXTENSION,
    'python': PYTHON_FILE_EXTENSION
}


def get_directories():
    return os.listdir(MAIN_RESOURCE_DIR)


def create_directory_for_search_phrase(directory_name):
    full_directory_path = f'../resources/{directory_name}'
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)


def get_python_single_line_comments(file):
    comments = []
    file_obj = open(file, 'r')
    for toktype, tok, start, end, line in tokenize.generate_tokens(file_obj.readline):
        # we can also use token.tok_name[toktype] instead of 'COMMENT'
        # from the token module
        if toktype == tokenize.COMMENT:
            print('COMMENT' + " " + tok)
            comments.append(tok)
    return comments


def find_range(comment_block, content, extension):
    print('finding comment start and end line in source code...')


def parse_file_comments(filename, file_content, language):
    file_comments = []
    if extension_mapping.get(language) == JAVA_FILE_EXTENSION:
        file_comments = re.findall(r'(?://[^\n]*|/\*(?:(?!\*/).)*\*/)', file_content, re.DOTALL)
    elif extension_mapping.get(language) == PYTHON_FILE_EXTENSION:
        file_comments = re.findall(r'([^:]"""[^\(]*)"""', file_content, re.DOTALL) + get_python_single_line_comments(filename)

    for comment in file_comments:
        comment_line_range = find_range(comment, file_content)


def parse_files(dir, language):
    file_contents = []
    file_names = []
    for path in pathlib.Path(f'{MAIN_RESOURCE_DIR}/{dir}').iterdir():
        if path.is_file() and extension_mapping.get(language) == path.suffix.format():
            current_file = open(path, "r")
            print(path)
            file_contents.append(current_file.read())
            file_names.append(path)
            current_file.close()
    return file_names, file_contents


if __name__ == '__main__':
    programming_language = input('Enter programming language for code parsing: ')
    directories = get_directories()
    for directory in directories:
        file_names, file_contents = parse_files(directory, programming_language.lower())
        for idx, file_content in enumerate(file_contents):
            comments = parse_file_comments(file_names[idx], file_content, programming_language.lower())
            #write_comments_to_file(comments)
