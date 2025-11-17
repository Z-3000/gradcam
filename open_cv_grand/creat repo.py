import requests

GITHUB_USER = "JH3907"
REPO_NAME = "xraygradcam"
TOKEN = "YOUR_GITHUB_TOKEN"  # ← 여기에 GitHub PAT 입력

url = f"https://api.github.com/user/repos"
headers = {"Authorization": f"token {TOKEN}"}
data = {"name": REPO_NAME, "private": False}

r = requests.post(url, headers=headers, json=data)
print(r.status_code, r.text)
