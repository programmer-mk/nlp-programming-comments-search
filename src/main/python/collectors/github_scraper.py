from helpers import string_helper, file_helper
import programming_languages_file_extensions as file_extensions
import requests
import os
from collections import namedtuple

PROGRAMMING_LANGUAGES_GITHUB_API_NAMES_MAPPING = {
    'java': 'java',
    'python': 'python',
    'c#' : 'csharp'
}

GITHUB_AUTH_TOKEN = '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'
GithubItem = namedtuple('GithubItem', 'file_name file_url repository_name repository_owner')

def download_files(github_files, search_phrase, extension):
    if len(github_files) > 0:
        directory_name = string_helper.split_string_with_underscore(search_phrase)
        file_helper.create_directory(directory_name)
        for github_file in github_files:
            url = github_file.file_url
            headers = {'accept': 'application/vnd.github.VERSION.raw'}
            response = requests.get(url, headers = headers)
            directory_to_save = f'resources/{directory_name}'
            file_to_save = github_file.file_name
            if os.path.exists(f'{directory_to_save}/{file_to_save}'):
                file_to_save = file_helper.append_guid_to_file_name(file_to_save)
            file_helper.write_bytes_to_file(directory_to_save, file_to_save, response.content)
    else:
        print('\nNo file was downloaded')

def get_language_name_for_github_api(programming_language):
    output = PROGRAMMING_LANGUAGES_GITHUB_API_NAMES_MAPPING.get(programming_language.lower())
    return output

def search_github(phrase, programming_language):
    output = []
    prog_language_github_api = get_language_name_for_github_api(programming_language)

    if prog_language_github_api == None:
        print(f"\n'{programming_language}' is not supported for code search")
        return output

    query_url = f"https://api.github.com/search/code?q={phrase}+in:file+language:{prog_language_github_api}"
    params = { "state": "open" }
    headers = {'Authorization': f'token {GITHUB_AUTH_TOKEN}'}

    response = requests.get(query_url, headers = headers, params = params)

    if(response.status_code == 200):
        responseJson = response.json()
        result_items = responseJson['items']
        output = [GithubItem(t['name'], t['git_url'], t['repository']['full_name'], t['repository']['owner']['id']) for t in result_items]
        print(f'\n{len(output)} results found in {programming_language} for a search criterion \'{phrase}\'')
    else:
        print(f'\nThere was an error while processing an HTTP request for a url: {query_url}')
        print(f'Reason: {response.reason}, Status code = {response.status_code}')

    return output

def write_success_message(natural_language, programming_language, search_phrase):
    print()
    print('Data has been successfully pulled from Github')
    print(f'Natural language: {natural_language}')
    print(f'Programming language: {programming_language}')
    print(f'Content searched: {search_phrase}')
    print()          

if __name__ == '__main__':
    print()
    natural_language = input('Enter a natural language for code search: ')
    search_phrase = input(f'Enter a search criterion in the {natural_language} language: ')
    programming_language = input('Enter a programming language for code search: ')
    extension = file_extensions.get_extension_for_programming_language(programming_language.lower())
    if extension == None:
        print()
        print(f'{programming_language} is not a supported programming language')
    else:
        files = search_github(search_phrase, programming_language)
        download_files(files, search_phrase, extension)
        write_success_message(natural_language, programming_language, search_phrase)
