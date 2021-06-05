import os
import pathlib
import re
import tokenize
import csv
from collections import namedtuple

MAIN_RESOURCE_DIR = '../../resources'
JAVA_FILE_EXTENSION = '.java'
PYTHON_FILE_EXTENSION = '.py'

# TODO: add C# here
extension_mapping = {
    'java': JAVA_FILE_EXTENSION,
    'python': PYTHON_FILE_EXTENSION
}

Comment = namedtuple('Comment', 'comment_text start_line end_line')


def get_directories():
    return os.listdir(MAIN_RESOURCE_DIR)


def create_directory_for_search_phrase(directory_name):
    full_directory_path = f'../resources/{directory_name}'
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)

    full_directory_path = f'../resources/{directory_name}/processed_comments'
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)


def write_comments_to_file(comments, file_path):
    print(f'start writing comments from {file_path}')
    with open(f'{file_path.parent}/processed_comments/{file_path.stem}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['comment_text', 'start_line', 'end_line']
        writer.writerow(header)
        for comment in comments:
            data = [comment.comment_text, comment.start_line, comment.end_line]
            writer.writerow(data)
    f.close()

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


def find_comment(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append(line_number)
    return list_of_results


def find_range(comment_block, filename, content, language):
    print(f'finding {language} comment start and end line in source code...')
    if extension_mapping.get(language) == JAVA_FILE_EXTENSION:
        if comment_block.startswith('//'):
            print('one line java comment')
            start_line = find_comment(filename,comment_block)
            return Comment(comment_block, start_line, start_line)
        elif comment_block.startswith('/*'):
            print('multiline java comment')
            start_line = find_comment(filename,comment_block[4:20])
            end_line = find_comment(filename,comment_block[len(comment_block)-20:len(comment_block)-4])
            return Comment(comment_block, start_line, end_line)
    elif extension_mapping.get(language) == PYTHON_FILE_EXTENSION:
        print('start python analyzing...')
        # TODO: same for python like java


def parse_file_comments(filename, file_content, language):
    file_comments = []
    if extension_mapping.get(language) == JAVA_FILE_EXTENSION:
        file_comments = re.findall(r'(?://[^\n]*|/\*(?:(?!\*/).)*\*/)', file_content, re.DOTALL)
    elif extension_mapping.get(language) == PYTHON_FILE_EXTENSION:
        file_comments = re.findall(r'([^:]"""[^\(]*)"""', file_content, re.DOTALL) + get_python_single_line_comments(filename)

    comments_full_info = []
    for comment in file_comments:
        comment_info = find_range(comment, filename, file_content, language)
        comments_full_info.append(comment_info)
    return comments_full_info


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
        create_directory_for_search_phrase(directory)
        file_names, file_contents = parse_files(directory, programming_language.lower())
        for idx, file_content in enumerate(file_contents):
            comments = parse_file_comments(file_names[idx], file_content, programming_language.lower())
            print('write comments to file')
            write_comments_to_file(comments, file_names[idx])
