import os

# 使用物理 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 必须在导入 vLLM 前设置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import gc
import json
from pathlib import Path

import torch
from prompt_illegal_building import ILLEGAL_BUILDING_PROMPT
from prompt_traffic_sign_defect import TRAFFIC_SIGN_PROMPT
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ============================================================
# Configuration
# ============================================================

PROMPT_MAP = {
    "illegal_building": ILLEGAL_BUILDING_PROMPT,
    "traffic_sign": TRAFFIC_SIGN_PROMPT,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# Input preparation
# ============================================================


def prepare_inputs_for_vllm(messages, processor):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    multi_modal_data = {}

    if image_inputs is not None:
        multi_modal_data["image"] = image_inputs

    if video_inputs is not None:
        multi_modal_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": video_kwargs,
    }


def build_messages(image_path: Path, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path.as_uri(),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


# ============================================================
# File collection
# ============================================================


def collect_images(input_path: Path, recursive=False):
    """
    支持：
    1. 单张图片
    2. 图片文件夹
    3. 递归扫描子文件夹
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片格式：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if recursive:
        candidates = input_path.rglob("*")
    else:
        candidates = input_path.iterdir()

    image_paths = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda x: str(x).lower(),
    )

    if not image_paths:
        raise RuntimeError(f"文件夹中没有找到图片：{input_path}")

    return image_paths


# ============================================================
# JSON parsing
# ============================================================


def extract_json(text: str):
    """
    尽量从模型输出中提取 JSON。

    支持：
    - 纯 JSON
    - ```json ... ```
    - JSON 前后带额外文字
    """
    text = text.strip()

    # 去掉 Markdown code fence
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # 情况 1：整个字符串就是 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 情况 2：前后存在模型解释文字
    decoder = json.JSONDecoder()

    for i, char in enumerate(text):
        if char not in "{[":
            continue

        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(f"模型输出中未找到合法 JSON：\n{text}")


# ============================================================
# vLLM
# ============================================================


def shutdown_vllm(llm):
    """
    尝试关闭 vLLM 后台进程，避免退出时出现 NCCL 警告。
    """
    if llm is None:
        return

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass

    del llm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Batch inference
# ============================================================


def run_batch(
    llm,
    processor,
    sampling_params,
    image_paths,
    prompt,
):
    """
    对一批图片进行推理。
    返回：
    [
        {
            "image": "...",
            "parse_success": True,
            "result": {...}
        }
    ]
    """
    inputs = []
    valid_paths = []
    results = []

    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    for image_path in image_paths:
        try:
            messages = build_messages(
                image_path=image_path,
                prompt=prompt,
            )

            vllm_input = prepare_inputs_for_vllm(
                messages=messages,
                processor=processor,
            )

            inputs.append(vllm_input)
            valid_paths.append(image_path)

        except Exception as exc:
            print(f"[输入失败] {image_path}")
            print(f"  {exc}")

            results.append(
                {
                    "image": str(image_path),
                    "parse_success": False,
                    "error": f"input_error: {exc}",
                }
            )

    if not inputs:
        return results

    # --------------------------------------------------------
    # vLLM inference
    # --------------------------------------------------------

    try:
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
        )

    except Exception as exc:
        # 如果整个 batch 推理失败，记录所有图片
        for image_path in valid_paths:
            results.append(
                {
                    "image": str(image_path),
                    "parse_success": False,
                    "error": f"inference_error: {exc}",
                }
            )

        return results

    # --------------------------------------------------------
    # Parse output
    # --------------------------------------------------------

    for image_path, output in zip(valid_paths, outputs):
        generated_text = output.outputs[0].text

        try:
            parsed = extract_json(generated_text)

            results.append(
                {
                    "image": str(image_path),
                    "parse_success": True,
                    "result": parsed,
                }
            )

        except (json.JSONDecodeError, ValueError) as exc:
            results.append(
                {
                    "image": str(image_path),
                    "parse_success": False,
                    "error": f"json_parse_error: {exc}",
                    "raw_output": generated_text,
                }
            )

    return results


def save_result(
    image_path: Path,
    result: dict,
    output_dir: Path,
):
    """
    每张图片保存一个独立 JSON。

    example:
        001.jpg -> 001.json
        abc.png -> abc.json
    """
    output_path = output_dir / f"{image_path.stem}.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen3-VL 批量分析图像")

    parser.add_argument(
        "--image",
        type=str,
        default=r"/scrinvme/huilin/IB/20260812_lands/image",
        help="单张图片或图片文件夹",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="JSON 结果输出文件夹",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL 模型名称或本地路径",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=PROMPT_MAP.keys(),
        default="illegal_building",
        help="识别任务",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="每批图片数量",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子文件夹",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    input_path = Path(args.image).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_paths = collect_images(
        input_path,
        recursive=args.recursive,
    )

    total = len(image_paths)

    print("=" * 80)
    print(f"任务       : {args.task}")
    print(f"模型       : {args.model}")
    print(f"输入路径   : {input_path}")
    print(f"图片数量   : {total}")
    print(f"Batch size : {args.batch_size}")
    print(f"输出目录   : {output_dir}")
    print("=" * 80)

    prompt = PROMPT_MAP[args.task]

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(args.model)

    # --------------------------------------------------------
    # vLLM
    # --------------------------------------------------------

    llm = None

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=16384,
            max_num_seqs=args.batch_size,
            limit_mm_per_prompt={
                "image": 1,
                "video": 0,
            },
            gpu_memory_utilization=0.90,
            seed=0,
            disable_log_stats=True,
        )

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            top_k=-1,
        )

        success_count = 0
        failed_count = 0

        # ----------------------------------------------------
        # Batch inference
        # ----------------------------------------------------

        for start in range(
            0,
            total,
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                total,
            )

            batch_paths = image_paths[start:end]

            print(f"\n[{start + 1}-{end} / {total}] 正在处理...")

            batch_results = run_batch(
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                image_paths=batch_paths,
                prompt=prompt,
            )

            # ------------------------------------------------
            # 每张图片立即保存一个 JSON
            # ------------------------------------------------

            for item in batch_results:
                image_path = Path(item["image"])

                if item["parse_success"]:
                    json_result = item["result"]
                    success_count += 1

                else:
                    json_result = {
                        "parse_success": False,
                        "error": item.get("error"),
                        "raw_output": item.get("raw_output"),
                    }
                    failed_count += 1

                output_path = save_result(
                    image_path=image_path,
                    result=json_result,
                    output_dir=output_dir,
                )

                print(f"  {image_path.name} -> {output_path.name}")

        print("\n" + "=" * 80)
        print("处理完成")
        print(f"总图片数：{total}")
        print(f"成功    ：{success_count}")
        print(f"失败    ：{failed_count}")
        print(f"结果目录：{output_dir}")
        print("=" * 80)

    finally:
        shutdown_vllm(llm)


if __name__ == "__main__":
    main()
