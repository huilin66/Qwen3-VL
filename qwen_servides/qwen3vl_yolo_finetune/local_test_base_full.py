import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

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
    """读取 Full LoRA 训练时使用的原始 prompt。"""
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    if not first_line:
        raise ValueError(f"Train JSONL is empty: {jsonl_path}")

    sample = json.loads(first_line)
    prompt = sample["messages"][0]["content"]

    if prompt.startswith("<image>"):
        prompt = prompt[len("<image>") :]

    return prompt.strip()


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    从模型输出中提取 JSON 数组。

    兼容：
    1. 纯 JSON 数组；
    2. Markdown JSON 代码块；
    3. JSON 前后存在少量额外文本。
    """
    text = text.strip()

    code_fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")

        if start < 0 or end < start:
            raise ValueError(f"No JSON array found in model output:\n{text}")

        result = json.loads(text[start : end + 1])

    if not isinstance(result, list):
        raise TypeError(f"Expected JSON list, got {type(result).__name__}")

    return result


def bbox_1000_to_pixels(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[int]:
    """把 Qwen3-VL 的 0～1000 坐标转换成原图像素坐标。"""
    if len(bbox) != 4:
        raise ValueError(f"bbox_2d must have 4 values: {bbox}")

    x1, y1, x2, y2 = map(float, bbox)

    pixel_bbox = [
        round(x1 / 1000.0 * image_width),
        round(y1 / 1000.0 * image_height),
        round(x2 / 1000.0 * image_width),
        round(y2 / 1000.0 * image_height),
    ]

    pixel_bbox[0] = max(0, min(image_width - 1, pixel_bbox[0]))
    pixel_bbox[1] = max(0, min(image_height - 1, pixel_bbox[1]))
    pixel_bbox[2] = max(0, min(image_width, pixel_bbox[2]))
    pixel_bbox[3] = max(0, min(image_height, pixel_bbox[3]))

    return pixel_bbox


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
    print("Task: base full detection")
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
        max_tokens=512,
        extra_body={
            "top_k": -1,
        },
    )

    content = response.choices[0].message.content

    print("Raw base detection output:")
    print("=" * 80)
    print(content)
    print("=" * 80)

    try:
        detections = extract_json_array(content)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"Failed to parse detection JSON: {error}")
        return

    with Image.open(IMAGE_PATH) as image:
        image_width, image_height = image.size

    parsed_results = []

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            print(f"Skip non-dict item {index}: {detection}")
            continue

        label = detection.get("label")
        bbox_2d = detection.get("bbox_2d")

        if label is None or bbox_2d is None:
            print(f"Skip incomplete item {index}: {detection}")
            continue

        try:
            bbox_pixels = bbox_1000_to_pixels(
                bbox=bbox_2d,
                image_width=image_width,
                image_height=image_height,
            )
        except (TypeError, ValueError) as error:
            print(f"Skip invalid bbox at index {index}: {error}")
            continue

        parsed_results.append(
            {
                "label": label,
                "bbox_1000": bbox_2d,
                "bbox_pixels": bbox_pixels,
            }
        )

    print("Parsed base detections:")
    print("=" * 80)
    print(
        json.dumps(
            parsed_results,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 80)

    if response.usage is not None:
        print("Token usage:")
        print(f"  prompt_tokens: {response.usage.prompt_tokens}")
        print(f"  completion_tokens: {response.usage.completion_tokens}")
        print(f"  total_tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
