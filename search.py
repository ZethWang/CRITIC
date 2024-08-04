

import os
import requests
from dotenv import find_dotenv,load_dotenv
_ = load_dotenv(find_dotenv())

def google_custom_search(api_key, cse_id, query, num_results=2):
    # Google Custom Search API 端点
    url = "https://www.googleapis.com/customsearch/v1"

    # 定义搜索参数
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'num': num_results
    }

    try:
        # 向 Google Custom Search API 发出请求
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果请求失败则引发异常
        search_results = response.json()
        return search_results

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP 错误
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")  # 连接错误
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")  # 超时错误
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")  # 请求错误

# Example usage
api_key = os.getenv('ERSPECTIVE_API_KEY') # Replace with your API key
cse_id = os.getenv('CSE_ID')    # Replace with your Custom Search Engine ID
query = "Python programming"

results = google_custom_search(api_key, cse_id, query)

# 显示搜索结果
if results:
    for item in results.get('items', []):
        print("标题:", item.get('title'))
        print("链接:", item.get('link'))
        print("摘要:", item.get('snippet'))
        print()
