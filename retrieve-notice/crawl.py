import re
import os
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

load_dotenv()

PAGE_OFFSET_STEP = 10
JSON_DIR = os.getenv("JSON_DIR")
PAGINATION_URL = os.getenv("PAGINATION_URL")
BASE_URL = os.getenv("BASE_URL")
HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}


def minify_text(text):
    result = re.sub(r"\s+", " ", text).replace("\r", "").strip()
    return result


def save_json(notice_id, title, content):
    data = {"title": title, "content": content}
    with open(f"{JSON_DIR}/{notice_id}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_pages_href(offset):
    url = PAGINATION_URL + str(offset)
    response = requests.get(url, headers=HEADER)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        all_links = soup.select("ul.board-list-wrap li a")
        all_href = [link.get("href") for link in all_links]
        return all_href
    else:
        print(f"Failed {offset}: {response.status_code}")
        return


def save_page(page_url):
    page_url = BASE_URL + page_url
    response = requests.get(page_url, headers=HEADER)
    parsed_url = urlparse(page_url)
    query_params = parse_qs(parsed_url.query)
    article_no = query_params.get("articleNo", [None])[0]
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        title = (
            soup.select_one(
                "#jwxe_main_content > div > div > div > div > table.board_view > thead > tr > th > em"
            )
            .get_text()
            .strip()
        )
        content = (
            soup.select_one(
                "#jwxe_main_content > div > div > div > div > table.board_view > tbody > tr > td > dl > dd > pre"
            )
            .get_text()
            .strip()
        )
        title = minify_text(title)
        content = minify_text(content)
        save_json(article_no, title, content)
    else:
        print(f"Failed {article_no}: {response.status_code}")


if __name__ == "__main__":
    for i in range(30):
        links = get_pages_href(i * PAGE_OFFSET_STEP)
        for link in links:
            save_page(link)
