# WD14-Batch

![2024-10-10_233502](https://github.com/user-attachments/assets/fbc5a92d-4b74-45a8-b72c-b3e76ec3a1b9)

~~面向 ChatGPT 编程的产物。~~

完全离线且本地的批量打标工具。原始项目：[Ketengan-Diffusion/wdv3-batch-vit-tagger](https://github.com/Ketengan-Diffusion/wdv3-batch-vit-tagger)

## 目录结构

```
├─input
├─model
│  ├─wd-eva02-large-tagger-v3
│  │      model.onnx
│  │      selected_tags.csv
│  │      
│  └─wd-vit-tagger-v3
│          model.onnx
│          selected_tags.csv
│          
└─python
│  ├─ ...
│  caption.py
│  requirements.txt
│  run.bat
│  run2.bat
```

## 安装

`requirements.txt` 使用 `pip freeze >> requirements.txt` 生成，所有包依赖指定了版本。

[安装参考](https://github.com/SGN-EARTH/JoyCaption-Pre-Alpha-Batch?tab=readme-ov-file#%E5%AE%89%E8%A3%85)，不用安装 torch 。或者按常规的 venv 虚拟环境或 conda 折腾。

## 获取模型

打开大佬[主页](https://huggingface.co/SmilingWolf)，选一个模型。

在 model 目录使用 git clone 拉取，或者打开对应模型地址，单独下载 model.onnx 和 selected_tags.csv 。

无特殊需求，使用 wd-eva02-large-tagger-v3 就够了。

```
# model.onnx 和 selected_tags.csv 一共 1.17 GB 。拉取会获取多余的文件，三四G左右。
git clone https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3
```

## 使用

把图片丢 input 目录中，运行 run2.bat 。

或者运行 run.bat 后执行 python caption.py -h 查看帮助。

> 以下仅供参考。不同的图片处理时间差异很小。
>
> 农企 5700x 占用百分之五十五附近，每一张图耗时两秒左右。使用小模型速度会更快。
>
> 牙膏厂 i5-9400F 系统自带壁纸会跑满 CPU ，每张图耗时接近三秒。
>
> 牙膏厂 i7-10750H 人物图最高百分之七十五，耗时四五秒。

## 自定义

如果需要修改使用特定模型，编辑 caption.py 把模型路径改成想要使用的模型所在的位置。

```
MODEL_DIR = "wd-eva02-large-tagger-v3"
或者
MODEL_DIR = "wd-vit-tagger-v3"
...
```

