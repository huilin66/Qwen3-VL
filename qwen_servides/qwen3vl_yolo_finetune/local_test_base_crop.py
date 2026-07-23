import json
from pathlib import Path

from openai import OpenAI

# ============================================================
# 服务配置
# ============================================================

BASE_URL = "http://127.0.0.1:8010/v1"
API_KEY = "qwenvlloraL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"

# 未加载 LoRA 的基座模型
MODEL_NAME = "qwen3-vl-4b-instruct"


# ============================================================
# 数据配置
# ============================================================

# 需要测试的目标裁剪图，而不是完整原图
IMAGE_PATH = Path(
    "/scrinvme/huilin/traffic_sign/defect/detection/data_det_3_damaged_traffic_signs/qwen_crop/crops/val/000_deformation/d0030_jpg.rf.a4f5e6523aba898f60b6620573ce0ee3_ac91f1b9a6__obj0000.jpg"
)

# Crop 训练数据转换后生成的 train.jsonl
TRAIN_JSONL = Path(
    "/scrinvme/huilin/traffic_sign/defect/detection/data_det_3_damaged_traffic_signs/qwen_crop/train.jsonl"
)


def load_training_prompt(jsonl_path: Path) -> str:
    """读取 Crop LoRA 训练时使用的原始 prompt。"""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    if not first_line:
        raise ValueError(f"Train JSONL is empty: {jsonl_path}")

    sample = json.loads(first_line)
    prompt = sample["messages"][0]["content"]

    # API 中图像已经单独传递
    if prompt.startswith("<image>"):
        prompt = prompt[len("<image>") :]

    return prompt.strip()


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    prompt = load_training_prompt(TRAIN_JSONL)
    image_url = IMAGE_PATH.resolve().as_uri()

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=300.0,
    )

    print(f"Model: {MODEL_NAME}")
    print("Task: base crop classification")
    print(f"Image: {IMAGE_PATH}")
    print(f"Prompt: {prompt}")
    print("=" * 80)

    response = client.chat.completions.create(
        model=MODEL_NAME,
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
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=32,
        extra_body={
            "top_k": -1,
        },
    )

    content = response.choices[0].message.content

    print("Base crop result:")
    print("=" * 80)
    print(content)
    print("=" * 80)

    if response.usage is not None:
        print("Token usage:")
        print(f"  prompt_tokens: {response.usage.prompt_tokens}")
        print(f"  completion_tokens: {response.usage.completion_tokens}")
        print(f"  total_tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
