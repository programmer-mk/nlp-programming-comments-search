from github import Github
GITHUB_AUTH_TOKEN = 'dummy'

if __name__ == '__main__':
    github_client = Github(GITHUB_AUTH_TOKEN)
    print(github_client.get_user().get_repos())
    # result should be similar <github.PaginatedList.PaginatedList object at ......>
    # if you got it then your setup in fine
