import os

def add_text_to_file_head(directory, text):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 确保只处理.txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # 读取原始文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                original_content = file.read()
            
            # 将指定文本和原始内容合并
            new_content = text + original_content
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
                
            print(f"Updated file: {file_path}")

# 示例使用
directory_path = 'input'
text_to_add = 'ysslz, yakushiji ryoko, '
add_text_to_file_head(directory_path, text_to_add)
