from helpers import string_helper, file_helper
import programming_languages_file_extensions as file_extensions
import os
import pathlib
import re
import tokenize
import csv
from collections import namedtuple

MAIN_RESOURCE_DIR = 'resources'

Comment = namedtuple('Comment', 'comment_text start_line end_line')

def get_directories():
    return os.listdir(MAIN_RESOURCE_DIR)

def write_comments_to_file(comments, file_path):

    print(f"\nWriting comments from a file called '{file_path}' started")

    with open(f'{file_path.parent}/processed_comments/{file_path.stem}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['comment_text', 'start_line', 'end_line']
        writer.writerow(header)
        for comment in comments:
            data = [comment.comment_text, comment.start_line, comment.end_line]
            # TODO: Dodati tab izmedju vrednosti
            writer.writerow(data)

    print(f"\nWriting comments from a file called '{file_path}' ended")

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
    file_extension = file_extensions.EXTENSION_MAPPING.get(language)
    if file_extension == file_extensions.JAVA_FILE_EXTENSION or file_extension == file_extensions.C_SHARP_FILE_EXTENSION:
        if comment_block.startswith('//'):
            start_line = find_comment(filename, comment_block)
            return Comment(comment_block, start_line, start_line)
        elif comment_block.startswith('/*'):
            start_line = find_comment(filename, comment_block[4:20])
            end_line = find_comment(filename, comment_block[len(comment_block)-20:len(comment_block)-4])
            return Comment(comment_block, start_line, end_line)

def find_range(comment_block, directory, filename, content, language):
    file_extension = file_extensions.EXTENSION_MAPPING.get(language)
    if file_extension == file_extensions.JAVA_FILE_EXTENSION or file_extension == file_extensions.C_SHARP_FILE_EXTENSION:
        if comment_block.startswith('//'):
            start_line = find_comment(filename, comment_block)
            print(comment_block)
            return Comment(comment_block, start_line, start_line)
        elif comment_block.startswith('/*'):
            print(comment_block)
            start_line = find_comment(filename, comment_block[4:20])
            end_line = find_comment(filename, comment_block[len(comment_block)-20:len(comment_block)-4])
            return Comment(comment_block, start_line, end_line)

def parse_file_comments(directory, filename, file_content, language):
    
    file_comments = []
    file_extension = file_extensions.EXTENSION_MAPPING.get(language)
     
    if file_extension != None:
        if file_extension == file_extensions.JAVA_FILE_EXTENSION or file_extension == file_extensions.C_SHARP_FILE_EXTENSION:
            file_comments = re.findall(r'(?://[^\n]*|/\*(?:(?!\*/).)*\*/)', file_content, re.DOTALL)

    comments_full_info = []

    for comment in file_comments:
        comment_info = find_range(comment, directory, filename, file_content, language)
        comments_full_info.append(comment_info)

    return comments_full_info

def parse_files(dir, language):

    file_contents = []
    file_names = []
    is_language_defined = language in file_extensions.EXTENSION_MAPPING.keys()

    print('\nParsing files started...\n')

    for path in pathlib.Path(f'{MAIN_RESOURCE_DIR}/{dir}').iterdir():

        is_file_extension_valid = file_extensions.EXTENSION_MAPPING.get(language) == path.suffix.format()
        
        if path.is_file() and is_language_defined and is_file_extension_valid:
            try:
                current_file = open(path, "r")
                file_contents.append(current_file.read())
                file_names.append(path)
                current_file.close()
                print(path)
            except Exception as e:
                print(f'Error - {e.__class__} - {path}')
        else:
            print(f'Error - Programming language specified or file extension not correct - {path}')

    print('\nParsing files ended\n')

    return file_names, file_contents

def write_success_message():
    print()
    print('All comments in all files are successfully parsed and writen to csv files')
    print()

if __name__ == '__main__':
    programming_language = input('Enter a programming language for code parsing: ')
    directories = get_directories()
    for directory in directories:
        file_helper.create_directory(directory)
        file_helper.create_directory(f'{directory}/processed_comments')
        file_names, file_contents = parse_files(directory, programming_language.lower())
        for idx, file_content in enumerate(file_contents):
            file_name = file_names[idx]
            comments = parse_file_comments(directory, file_name, file_content, programming_language.lower())
            write_comments_to_file(comments, file_name)
    write_success_message()
    
