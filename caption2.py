import os
import numpy as np
import pandas as pd
import onnxruntime as rt
from PIL import Image
import argparse
from pathlib import Path
import time

"""

# WD EVA02-Large Tagger v3 
https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3

# WD ViT Tagger v3 
https://huggingface.co/SmilingWolf/wd-vit-tagger-v3

"""

# 默认输入和输出文件夹路径
INPUT_FOLDER = Path(__file__).parent / "input"

MODEL_DIR = "wd-eva02-large-tagger-v3"
# MODEL_DIR = "wd-vit-tagger-v3"

MODEL_PATH = Path(f"model/{MODEL_DIR}/model.onnx")  # ONNX模型的本地路径
LABEL_PATH = Path(f"model/{MODEL_DIR}/selected_tags.csv")  # 标签的本地路径

# 加载模型和标签
def load_model_and_tags(use_gpu=False):
    print("加载模型")  # 加载模型提示
    start_time = time.time()
    tags_df = pd.read_csv(LABEL_PATH)
    tag_data = LabelData(
        names=tags_df["name"].tolist(),
        rating=list(np.where(tags_df["category"] == 9)[0]),
        general=list(np.where(tags_df["category"] == 0)[0]),
        character=list(np.where(tags_df["category"] == 4)[0]),
    )
    
    # 配置 ONNX Runtime 以使用 GPU 或 CPU
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    model = rt.InferenceSession(MODEL_PATH, providers=providers)
    target_size = model.get_inputs()[0].shape[2]
    load_time = time.time() - start_time
    print(f"已加载模型，耗时: {load_time:.2f} 秒。\n")
    return model, tag_data, target_size

# 图像预处理函数
def prepare_image(image, target_size):
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
    image = canvas.convert("RGB")

    # 将图像填充为正方形
    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # 调整大小
    padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    
    # 转换为numpy数组
    image_array = np.asarray(padded_image, dtype=np.float32)[..., [2, 1, 0]]
    return np.expand_dims(image_array, axis=0)  # 添加批次维度

class LabelData:
    def __init__(self, names, rating, general, character):
        self.names = names
        self.rating = rating
        self.general = general
        self.character = character

# 根据阈值处理预测结果
def process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first):
    scores = preds.flatten()
    
    character_tags = [tag_data.names[i] for i in tag_data.character if scores[i] >= character_thresh]
    general_tags = [tag_data.names[i] for i in tag_data.general if scores[i] >= general_thresh]
    rating_tags = [] if hide_rating_tags else [tag_data.names[i] for i in tag_data.rating]

    final_tags = character_tags + general_tags if character_tags_first else general_tags + character_tags
    final_tags += rating_tags  # 如果不隐藏，添加评分标签
    return final_tags

# 标记图像
def tag_images(image_folder, include_subdirectories=False, character_tags_first=False, general_thresh=0.35, character_thresh=0.85, hide_rating_tags=False, remove_separator=False, use_gpu=False):
    model, tag_data, target_size = load_model_and_tags(use_gpu)

    processed_files = []

    def process_folder(folder):
        nonlocal processed_files
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path) and item.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'webp')):
                print(f"处理图片：{item_path}")  # 开始处理图片的提示
                try:
                    with Image.open(item_path) as image:
                        start_time = time.time()
                        processed_image = prepare_image(image, target_size)
                        preds = model.run(None, {model.get_inputs()[0].name: processed_image})[0]
                        processing_time = time.time() - start_time
                        print(f"耗时: {processing_time:.2f} 秒。")
                except Exception as e:
                    print(f"Error processing {item_path}: {e}")
                    continue

                # 处理预测结果
                final_tags = process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first)

                final_tags_str = ", ".join(final_tags)
                if remove_separator:
                    final_tags_str = final_tags_str.replace("_", " ")

                # 将结果写入文本文件，保存到与图片相同的目录
                caption_file_path = os.path.join(folder, f"{os.path.splitext(item)[0]}.txt") # 修改为保存到当前图片所在目录
                with open(caption_file_path, 'w') as f:
                    f.write(final_tags_str)

                processed_files.append(item_path)
            elif include_subdirectories and os.path.isdir(item_path):
                process_folder(item_path)

    process_folder(image_folder)

    return "\n所有图片处理完成。\n", "\n".join(processed_files)

def main():
    parser = argparse.ArgumentParser(description='图像标记脚本')
    parser.add_argument('image_folder', nargs='?', type=str, default=str(INPUT_FOLDER), help='图像目录路径（默认："input" 文件夹）')
    parser.add_argument('--include_subdirectories', action='store_true', help='是否处理子目录')
    parser.add_argument('--character_tags_first', action='store_true', help='优先显示角色标签')
    parser.add_argument('--general_thresh', type=float, default=0.35, help='普通标签阈值')
    parser.add_argument('--character_thresh', type=float, default=0.85, help='角色标签阈值')
    parser.add_argument('--hide_rating_tags', action='store_true', help='隐藏评分标签')
    parser.add_argument('--remove_separator', action='store_true', help='移除标签中的分隔符')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU进行推理')

    args = parser.parse_args()

    status, processed_files = tag_images(
        args.image_folder,
        include_subdirectories=args.include_subdirectories,
        character_tags_first=args.character_tags_first,
        general_thresh=args.general_thresh,
        character_thresh=args.character_thresh,
        hide_rating_tags=args.hide_rating_tags,
        remove_separator=args.remove_separator,
        use_gpu=args.use_gpu
    )

    print(status)
    print("已处理的文件：")
    print(processed_files)

if __name__ == "__main__":
    main()
