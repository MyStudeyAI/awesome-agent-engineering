# test_api.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")

if not api_key:
    print("❌ 没读到 ZHIPUAI_API_KEY，检查 .env 文件")
    exit()

print("-" * 50)

# 1. 测试 paas/v4 端点 + glm-4-flash
def test_chat_paas():
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": "Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": "只回复：OK"}],
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        print("【1】paas/v4 + glm-4-flash")
        print("状态码:", r.status_code)
        print("返回:", r.text[:300])
    except Exception as e:
        print("【1】请求异常:", e)
    print("-" * 50)

# 2. 测试 anthropic 端点 + glm-4-flash
def test_chat_anthropic():
    url = "https://open.bigmodel.cn/api/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    data = {
        "model": "glm-4-flash",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "只回复：OK"}],
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        print("【2】anthropic + glm-4-flash")
        print("状态码:", r.status_code)
        print("返回:", r.text[:300])
    except Exception as e:
        print("【2】请求异常:", e)
    print("-" * 50)

# 3. 测试 embedding-3
def test_embedding():
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": "Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "embedding-3",
        "input": ["测试文本"],
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        print("【3】embedding-3")
        print("状态码:", r.status_code)
        print("返回:", r.text[:300])
    except Exception as e:
        print("【3】请求异常:", e)
    print("-" * 50)

if __name__ == "__main__":
    test_chat_paas()
    test_chat_anthropic()
    test_embedding()
