import os

from openai import OpenAI


API_KEY = "qwenvlL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key=API_KEY,
    timeout=300.0,
)

image_url = (
    "file:///scrinvme/huilin/exchange/qwen3_vl/"
    "test_images/DA5324655_20251006155244700.jpg"
)

response = client.chat.completions.create(
    model="qwen3-vl-8b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "请识别图像中是否存在交通标志牌，"
                        "并说明交通标志牌的类别和具体类型。"
                    ),
                },
            ],
        }
    ],
    temperature=0,
    max_tokens=1024,
    extra_body={
        "top_k": -1,
    },
)

content = response.choices[0].message.content

print("=" * 80)
print(content)
print("=" * 80)