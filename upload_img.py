import os
from github import Github
from github.GithubException import GithubException
from dotenv import load_dotenv
load_dotenv()

g = Github(os.getenv('GITHUB_ACCOUNT'), os.getenv('GITHUB_TOKEN'))

def upload_file_to_github(file_path):
    username = "SeanChenR"
    repo_name = "Your Repo"
    branch = "main"
    repo = g.get_user().get_repo(repo_name)
    with open(file_path, 'rb') as image:
        i = image.read()
        image_data = bytearray(i)

    try:
        repo.create_file(file_path, "commit image", bytes(image_data), branch=branch)
    except GithubException as e:
        if e.status == 422:  # 422 表示檔案已存在
            # 取得現有檔案的 sha 值，進行更新
            contents = repo.get_contents(file_path, ref=branch)
            repo.update_file(contents.path, "update image", bytes(image_data), contents.sha, branch=branch)
            print(f"File {file_path} updated successfully.")
        else:
            raise e

    url = f"https://raw.githubusercontent.com/{username}/{repo_name}/{branch}/{file_path}"
    print(f"GitHub URL : {url}")
    with open("backtest_img_url.txt", "a") as f:
        f.write(url + '\n')
