import os
import requests

def test_openai():
    assert "OPENAI_API_KEY" in os.environ
    API_KEY = os.environ["OPENAI_API_KEY"]
    headers = {
        "Authorization": f'Bearer {API_KEY}'
    }
    endpoint = "https://api.openai.com/v1/engines"
    res = requests.get(endpoint, headers=headers)
    assert res.status_code == 200