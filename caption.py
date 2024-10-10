import os
import numpy as np
import pandas as pd
import onnxruntime as rt
from PIL import Image
import argparse
from pathlib import Path
import time

# 默认输入和输出文件夹路径
INPUT_FOLDER = Path(__file__).parent / "input"
OUTPUT_FOLDER = INPUT_FOLDER
# MODEL_DIR = "wd-vit-tagger-v3"
MODEL_DIR = "wd-eva02-large-tagger-v3"
MODEL_PATH = Path(f"model/{MODEL_DIR}/model.onnx")  # ONNX模型的本地路径
LABEL_PATH = Path(f"model/{MODEL_DIR}/selected_tags.csv")  # 标签的本地路径

# 加载模型和标签
def load_model_and_tags():
    print("加载模型")  # 加载模型提示
    start_time = time.time()
    tags_df = pd.read_csv(LABEL_PATH)
    tag_data = LabelData(
        names=tags_df["name"].tolist(),
        rating=list(np.where(tags_df["category"] == 9)[0]),
        general=list(np.where(tags_df["category"] == 0)[0]),
        character=list(np.where(tags_df["category"] == 4)[0]),
    )
    model = rt.InferenceSession(MODEL_PATH)
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
def tag_images(image_folder, character_tags_first=False, general_thresh=0.35, character_thresh=0.85, hide_rating_tags=False, remove_separator=False):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model, tag_data, target_size = load_model_and_tags()
    
    processed_files = []
    
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'webp')):
            print(f"处理图片：{image_file}")  # 开始处理图片的提示
            image_path = os.path.join(image_folder, image_file)
            with Image.open(image_path) as image:
                start_time = time.time()
                processed_image = prepare_image(image, target_size)
                preds = model.run(None, {model.get_inputs()[0].name: processed_image})[0]
                processing_time = time.time() - start_time
                print(f"耗时: {processing_time:.2f} 秒。")

            # 处理预测结果
            final_tags = process_predictions_with_thresholds(preds, tag_data, character_thresh, general_thresh, hide_rating_tags, character_tags_first)
            
            final_tags_str = ", ".join(final_tags)
            if remove_separator:
                final_tags_str = final_tags_str.replace("_", " ")

            # 将结果写入文本文件
            caption_file_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(image_file)[0]}.txt")
            with open(caption_file_path, 'w') as f:
                f.write(final_tags_str)

            processed_files.append(image_file)

    return "\n所有图片处理完成。\n", "\n".join(processed_files)

def main():
    parser = argparse.ArgumentParser(description='图像标记脚本')
    parser.add_argument('image_folder', nargs='?', type=str, default=str(INPUT_FOLDER), help='图像目录路径（默认："input" 文件夹）')
    parser.add_argument('--character_tags_first', action='store_true', help='优先显示角色标签')
    parser.add_argument('--general_thresh', type=float, default=0.35, help='普通标签阈值')
    parser.add_argument('--character_thresh', type=float, default=0.85, help='角色标签阈值')
    parser.add_argument('--hide_rating_tags', action='store_true', help='隐藏评分标签')
    parser.add_argument('--remove_separator', action='store_true', help='移除标签中的分隔符')

    args = parser.parse_args()

    status, processed_files = tag_images(
        args.image_folder,
        character_tags_first=args.character_tags_first,
        general_thresh=args.general_thresh,
        character_thresh=args.character_thresh,
        hide_rating_tags=args.hide_rating_tags,
        remove_separator=args.remove_separator
    )
    
    print(status)
    print("已处理的文件：")
    print(processed_files)

if __name__ == "__main__":
    main()
