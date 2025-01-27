import time
import pprint
import sys,os
# 获取当前脚本的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 获取到 llm-agent-web-tools/src/tools 的路径
# module_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'llm-agent-web-tools', 'src', 'tools','web_tools','core','engines'))

# # 将模块路径添加到 sys.path
# sys.path.append(module_path)

# 导入 search_api 模块
from google import Search as google_search
# TODO: upload google API & web page parser
# google_search = None
bing_search = None
wiki_search = None


def google(query, cache=True, topk=1, end_year=None, verbose=False):
    assert topk >= 1

    gresults = {"page": None, "title": None}

    trial = 0
    while gresults['page'] is None and trial < 3:
        trial += 1
        if trial > 1:
            print("Search Fail, Try again...")
        gresults = google_search(query, cache=cache, topk=topk, end_year=end_year)
        time.sleep(3 * trial)

    if verbose:
        pprint.pprint(gresults)

    return gresults


def _test_google():

    queries = [
        "The answer to life, the universe, and everything?"
    ]

    for q in queries:
        for topk in range(1, 2):
            res = google(q, verbose=True, cache=False, topk=topk)
            print(f"[{res.get('title', '')}] {res.get('page', '')}")


if __name__ == "__main__":
    _test_google()