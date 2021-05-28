import os
import requests
from collections import namedtuple

GITHUB_AUTH_TOKEN = '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'
GithubItem = namedtuple('GithubItem', 'file_name file_url repository_name repository_owner')


def compute_directory_name(phrase):
    terms = phrase.split()
    return '_'. join(terms)


def create_directory_for_search_phrase(directory_name):
    full_directory_path = f'../resources/{directory_name}'
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)


def download_files(github_files, search_phrase):
    directory_name = compute_directory_name(search_phrase)
    create_directory_for_search_phrase(directory_name)
    for github_file in github_files:
        url = github_file.file_url
        headers = {'accept': 'application/vnd.github.VERSION.raw'}
        r = requests.get(url, headers=headers)
        open(f'../resources/{directory_name}/{github_file.file_name}', 'wb').write(r.content)


def remove_dir_from_name(path):
    startIndex = path.rfind('/')
    file_name = path[startIndex+1: len(path)]
    return file_name


def search_github(phrase, prog_language):
    query_url = f"https://api.github.com/repos/0shade0/RP3/git/trees/master?recursive=1"
    params = {
        "state": "open",
    }
    headers = {'Authorization': f'token {GITHUB_AUTH_TOKEN}'}
    r = requests.get(query_url, headers=headers, params=params)
    result_items = r.json()['tree']
    items = [GithubItem(remove_dir_from_name(t['path']), t['url'], 'empty', 'empty') for t in result_items if t['path'].endswith('.cs') ]
    return items


if __name__ == '__main__':
    natural_language = input('Enter natural language for code searching: ')
    search_phrase = input(f'Enter search phrase on {natural_language} language: ')
    programming_language = input('Enter programming language for code searching: ')
    files = search_github(search_phrase, programming_language)
    download_files(files, search_phrase)
