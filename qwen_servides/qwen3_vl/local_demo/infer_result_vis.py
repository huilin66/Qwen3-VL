import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# =========================
# 配置
# =========================

JSON_DIR = Path("./results")
IMAGE_DIR = Path("/scrinvme/huilin/IB/20260812_lands/image")
OUTPUT_DIR = Path("./vis")

# Qwen 输出 bbox 的坐标范围
COORD_MAX = 1000

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


def get_font(size=24):
    """获取支持中文的字体。"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]

    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def find_image(image_dir, stem):
    """根据 JSON 文件名寻找对应图片。"""
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

        # 兼容大写扩展名
        image_path = image_dir / f"{stem}{ext.upper()}"

        if image_path.exists():
            return image_path

    return None


def load_json(json_path):
    """读取 JSON 文件。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_bbox(bbox, width, height):
    """
    将 Qwen 0~1000 bbox 转换为真实图像坐标。

    bbox:
        [x_min, y_min, x_max, y_max]
    """

    x1, y1, x2, y2 = bbox

    x1 = int(x1 / COORD_MAX * width)
    y1 = int(y1 / COORD_MAX * height)
    x2 = int(x2 / COORD_MAX * width)
    y2 = int(y2 / COORD_MAX * height)

    # 防止坐标越界
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def draw_detection(draw, obj, image_size, font):
    """绘制单个检测结果。"""

    width, height = image_size

    bbox = obj.get("bounding_box")

    if bbox is None or len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = convert_bbox(
            bbox,
            width,
            height,
        )
    except Exception:
        return False

    # 防止错误 bbox
    if x2 <= x1 or y2 <= y1:
        return False

    category = obj.get(
        "category",
        "疑似构筑物",
    )

    confidence = obj.get(
        "confidence",
        None,
    )

    obj_id = obj.get(
        "id",
        "",
    )

    # 标签
    if confidence is not None:
        try:
            label = f"{obj_id} {category} {float(confidence):.2f}"
        except Exception:
            label = f"{obj_id} {category} {confidence}"
    else:
        label = f"{obj_id} {category}"

    # bbox
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=4,
    )

    # 标签大小
    text_bbox = draw.textbbox(
        (0, 0),
        label,
        font=font,
    )

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 默认放在框上方
    label_x = x1
    label_y = y1 - text_height - 10

    # 上方空间不够
    if label_y < 0:
        label_y = y1 + 5

    # 防止标签超出右边界
    if label_x + text_width + 10 > width:
        label_x = max(
            0,
            width - text_width - 10,
        )

    # 标签背景
    draw.rectangle(
        [
            label_x,
            label_y,
            label_x + text_width + 10,
            label_y + text_height + 8,
        ],
        fill="red",
    )

    # 标签文字
    draw.text(
        (
            label_x + 5,
            label_y + 3,
        ),
        label,
        fill="white",
        font=font,
    )

    return True


def visualize_one(
    image_path,
    json_path,
    output_path,
):
    """可视化一张图片。"""

    result = load_json(json_path)

    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    # 根据图片尺寸自适应字体
    font_size = max(
        16,
        int(min(image.size) * 0.025),
    )

    font = get_font(font_size)

    instances = result.get(
        "instances",
        [],
    )

    valid_count = 0

    for obj in instances:
        success = draw_detection(
            draw,
            obj,
            image.size,
            font,
        )

        if success:
            valid_count += 1

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path,
        quality=95,
    )

    return valid_count


def batch_visualize(
    json_dir,
    image_dir,
    output_dir,
):
    """批量可视化。"""

    json_files = sorted(json_dir.glob("*.json"))

    print(f"找到 {len(json_files)} 个 JSON 文件")

    success_count = 0
    failed_count = 0

    for index, json_path in enumerate(
        json_files,
        1,
    ):
        stem = json_path.stem

        image_path = find_image(
            image_dir,
            stem,
        )

        if image_path is None:
            print(f"[{index}/{len(json_files)}] [跳过] 找不到图片: {stem}")

            failed_count += 1
            continue

        # 保留原图扩展名
        output_path = output_dir / image_path.name

        try:
            count = visualize_one(
                image_path,
                json_path,
                output_path,
            )

            print(f"[{index}/{len(json_files)}] {image_path.name} -> {count} 个目标")

            success_count += 1

        except Exception as e:
            print(f"[{index}/{len(json_files)}] [失败] {stem}: {e}")

            failed_count += 1

    print()
    print("=" * 50)
    print(f"完成: {success_count}")
    print(f"失败/跳过: {failed_count}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_visualize(
        json_dir=JSON_DIR,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
    )
