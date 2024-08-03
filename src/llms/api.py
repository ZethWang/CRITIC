import os
import json
import time
import pprint
import requests
from openai import OpenAI
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv,find_dotenv

# openai.api_key = ""
# openai.api_key = os.environ["OPENAI_API_KEY"]
_ =load_dotenv(find_dotenv())
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

def llm(prompt, model="glm-4-flash"
, temperature=0, stop=None, logprobs=None, n=1, max_tokens=512, verbose=False):
    
    client = OpenAI(api_key=ZHIPUAI_API_KEY,
                    base_url="https://open.bigmodel.cn/api/paas/v4/")
    for trial in range(3):
        if trial > 0:
            time.sleep(trial * 2)
        try:
            # set parameters
            parameters = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stop": stop,
                "temperature": temperature,
                "n": n,
                "logprobs": logprobs
            }
            # parameters.update({"messages": pprompt})
            resp = client.chat.completions.create(**parameters)
            if verbose:
                print(resp)
            text = resp.choices[0].message
            print(text.content)
            assert len(text.content) > 0
        except BaseException as e:
            print(">" * 20, "LLMs error", e)
            resp = None
        if resp is not None:
            break
    return resp


def _test_llm():
    model = "glm-4-flash"

    prompt = "Q: American Callan Pinckney’s eponymously named system became a best-selling (1980s-2000s) book/video franchise in what genre?"
    prompt += "A: "
    llm(prompt, model, stop=["\n"], max_tokens=20, verbose=True,logprobs=4,n=3)


if __name__ == "__main__":
    _test_llm()
   