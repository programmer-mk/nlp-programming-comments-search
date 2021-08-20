import requests
import pandas as pd
import sys
GITHUB_AUTH_TOKEN = 'dummy'

operating_system = sys.platform

VALIDATE_COUNT = 100

resources_directory = '../../resources'
if(operating_system == 'win32'):
    resources_directory = 'src\main/resources'


def check_url_existence(url):
    headers = {'accept': 'application/vnd.github.VERSION.raw'}
    response = requests.get(url, headers = headers)
    if(response.status_code != 200):
        return False
    else:
        return True


def validate_file_urls():
    data_frame_global = pd.read_excel(f'{resources_directory}/programming_comments_annotation.xlsx')
    data_frame_global.dropna(how='any', inplace=True)
    critical_comments_df = data_frame_global[100:240]
    source_urls = critical_comments_df["SourceDescription"].tolist()
    for url in source_urls:
        response = check_url_existence(url)
        if not response:
            print(f' not good url: {url}')
        else:
            print(f'good url: {url}')

if __name__ == '__main__':
    validate_file_urls()
