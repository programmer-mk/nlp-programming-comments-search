import requests
from collections import namedtuple

GITHUB_AUTH_TOKEN = 'dummy'
GithubItem = namedtuple('GithubItem', 'file_name file_url repository_name repository_owner')


def search_github(phrase, prog_language):
    query_url = f"https://api.github.com/search/code?q={phrase}+in:file+language:{prog_language}"
    params = {
        "state": "open",
    }
    headers = {'Authorization': f'token {GITHUB_AUTH_TOKEN}'}
    r = requests.get(query_url, headers=headers, params=params)
    result_items = r.json()['items']
    items = [GithubItem(t['name'], t['html_url'], t['repository']['full_name'], t['repository']['owner']['id']) for t in result_items]
    return items


if __name__ == '__main__':
    natural_language = input('Enter natural language for code searching: ')
    search_phrase = input(f'Enter search phrase on {natural_language} language: ')
    programming_language = input('Enter programming language for code searching: ')
    files = search_github(search_phrase, programming_language)
