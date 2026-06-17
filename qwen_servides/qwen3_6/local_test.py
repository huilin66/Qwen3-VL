import os
import sys

from openai import APIConnectionError, APIStatusError, OpenAI


BASE_URL = "http://127.0.0.1:8002/v1"
MODEL_NAME = "qwen3.6"


def main() -> None:
    api_key = "qwenL666666qB9pM8yVZx2Kf7R4hN1uCe6Wsd3AjT0mGkPqX"

    if not api_key:
        raise RuntimeError(
            "未设置环境变量 QWEN36_API_KEY。\n"
            "请先执行：\n"
            "export QWEN36_API_KEY='你的API_KEY'"
        )

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=300.0,
        max_retries=0,
    )

    print(f"服务地址：{BASE_URL}")
    print(f"请求模型：{MODEL_NAME}")

    # 1. 检查模型列表
    print("\n正在检查模型列表……")

    models = client.models.list()
    model_ids = [model.id for model in models.data]

    print(f"服务器模型：{model_ids}")

    if MODEL_NAME not in model_ids:
        raise RuntimeError(
            f"服务器中未找到模型 {MODEL_NAME}，"
            f"当前模型列表为：{model_ids}"
        )

    # 2. 发起聊天请求
    print("\n正在发送聊天请求……")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是建筑与交通设施检测系统中的分析助手。"
                    "回答应准确、简洁，不确定的信息需要明确说明。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请简要说明交通标志牌出现褪色后，"
                    "可能造成的安全风险以及建议的处置措施。"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    message = response.choices[0].message

    print("\n" + "=" * 80)
    print("模型回答：")
    print(message.content)
    print("=" * 80)

    # 部分 reasoning 模型可能把思考内容放在额外字段中
    reasoning_content = None

    if getattr(message, "model_extra", None):
        reasoning_content = message.model_extra.get("reasoning_content")

    if reasoning_content:
        print("\n模型推理内容：")
        print("-" * 80)
        print(reasoning_content)
        print("-" * 80)

    if response.usage:
        print("\nToken 使用情况：")
        print(f"输入 Token：{response.usage.prompt_tokens}")
        print(f"输出 Token：{response.usage.completion_tokens}")
        print(f"总 Token：{response.usage.total_tokens}")


if __name__ == "__main__":
    try:
        main()
    except APIConnectionError as exc:
        print(
            "\n无法连接 Qwen3.6 服务。\n"
            "请确认 vLLM 已启动，并监听 127.0.0.1:8002。"
        )
        print(f"详细错误：{exc}")
        sys.exit(1)
    except APIStatusError as exc:
        print(f"\n服务器返回 HTTP {exc.status_code}")
        print(f"错误内容：{exc.response.text}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n测试失败：{type(exc).__name__}: {exc}")
        sys.exit(1)