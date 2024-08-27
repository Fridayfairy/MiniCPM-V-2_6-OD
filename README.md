本教程是基于开源的MiniCPM-V 2.6在自定义数据集中做目标检测任务的微调，其他类似的下游任务同样可以参考本教程进行微调。

GitHub开源地址：https://github.com/Fridayfairy/MiniCPM-V-2_6-OD

> 作者：杨崇旭
> 
> 日期：2024.08

# 1. 数据处理
---


## 1.1 数据准备

整理原始图片数据集，并使用labelimg、labelme等标注工具，获得xml格式的标注文件。将图片放在images路径下，标注文件放在Annotations文件下。

## 1.2 数据处理

### 1.2.1 格式转换，执行xml2cpm26.py

本脚本作用是读取xml文件，将它转换成CPM所需要的json格式。

示例中，读取的路径是：./dataset/Annotations/' # 替换为你的XML文件所在目录，

```txt
<annotation>
    <folder>images</folder>
    <filename>demo.jpg</filename>
    <path>./images/demo.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>2592</width>
        <height>1944</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>wajueji</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>530</xmin>
            <ymin>1248</ymin>
            <xmax>1298</xmax>
            <ymax>1740</ymax>
        </bndbox>
    </object>
    <object>
        <name>wajueji</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>569</xmin>
            <ymin>1103</ymin>
            <xmax>1200</xmax>
            <ymax>1505</ymax>
        </bndbox>
    </object>
</annotation>
```

生成路径是：./dataset/dataset.json

```JSON
{
    "data": {
        "/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/images/demo.jpg": {
            "gt": [
                {
                    "text": "挖掘机",
                    "bbox": [
                        204,
                        641,
                        500,
                        895
                    ]
                },
                {
                    "text": "挖掘机",
                    "bbox": [
                        219,
                        567,
                        462,
                        774
                    ]
                }
            ],
            "image_path": "/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/images/demo.jpg"
        }
    }
}
```

### 1.2.2 数据预处理，执行preprocess.py

本脚本主要是将bbox转成CPM的对话文本。

核心步骤是将bbox的左上角坐标(x1, y1),右下角坐标(x2,y2),使用<box></box>标签包裹，再接上提示文本text，组成CPM对话的answer部分。

```Python
text = "请识别图像内的挖掘机，并附上它们的坐标。"
format_answer += "<ref>{text}<box>{x1}</box><box>{y1}</box><box>{x2}</box><box>{y2}</box></ref>".format(
            text=textes[index], x1=x1, y1=y1, x2=x2, y2=y2
        )
```

生成的json格式如下：

```JSON
[
    {
        "id": "0",
        "image": "/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/images/demo.jpg",
        "conversations": [
            {
                "content": "<image>\nPlease help me identify the construction machinery in the picture, and mark their exact locations.",
                "role": "user"
            },
            {
                "content": "<ref>挖掘机<box>204</box><box>641</box><box>500</box><box>895</box></ref><ref>挖掘机<box>219</box><box>567</box><box>462</box><box>774</box></ref>",
                "role": "assistant"
            }
        ]
    }
]
```

## 1.3 处理完成

处理完成后的dataset目录下文件分布如下，大家可以根据具体项目动态调整文件结构

![](https://bq82ox0och.feishu.cn/space/api/box/stream/download/asynccode/?code=YjgxYjVjZDA2YjNmZjYyMjAyNDJiNjc2NDk3OTRkMjVfWm5qMGY4V3hZUWw3WlNBNXgwRzZZMThrcmIyM1N6TFhfVG9rZW46UUI5YWI0RFpyb3dzdzl4OERvWGMyd3BnbktmXzE3MjQ3MzcxNTI6MTcyNDc0MDc1Ml9WNA)

# 2. 任务微调
---


在finetune文件下，修改finetune_lora.sh文件中项目路径。其他lora参数视训练环境进行调整。

```bash
MODEL="/data1/home/ycx/workspace/HF/MiniCPM-V-2_6" # 从HuggingFace 或者魔塔下载的官方 MiniCPM-V-2_6路径
DATA="/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/train_set.json" # 训练集路径
EVAL_DATA="/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/test_set.json" # 测试集路径
```

执行：

```Shell
bash finetune_lora.sh
```

# 3. 模型合并
---


在finetune文件下，修改merge_lora.py文件中项目路径。

```Shell
model_type="/data1/home/ycx/workspace/HF/MiniCPM-V-2_6" # 官方下载的原始模型
path_to_adapter="/data1/home/ycx/ycxGit/MiniCPM-V26/finetune/output/output__lora/checkpoint-10000" # lora保存的地址
merge_path="./MiniCPM-V-2_6-OD" # 希望将lora合并到主模型后的保存地址
```

注意检查合并后路径下是否缺少文件，本脚本会自动复制下列文件

```Shell
# 将缺少的文件复制到新模型路径下
copy_files = ['image_processing_minicpmv.py', 'preprocessor_config.json', 'processing_minicpmv.py']
```

# 4. 效果展示
---


使用web服务启动server，选择图片，提示：“执行目标检测任务”，

输出：<ref>挖掘机<box>197</box><box>582</box><box>430</box><box>860</box></ref>

后台数据显示,后面就可以通过提取answer中的坐标进行后端的处理了：

```Shell
msgs: [{'role': 'user', 'content': [<PIL.Image.Image image mode=RGB size=2592x1944 at 0x7F4BA42C6B90>, '执行目标检测任务']}]
answer:
<ref>挖掘机<box>197</box><box>582</box><box>430</box><box>860</box></ref>
```

可视化展示：

![](https://bq82ox0och.feishu.cn/space/api/box/stream/download/asynccode/?code=Yjk0MjcwZjZiNzEyZmE0OTliZmI3YjVhMDIzYjQyZTZfSlpEOHdWeGY2eElVV0s2akNzNDJZWUUyVjBnQmwyMGxfVG9rZW46VVNWVWJUd2hOb0ttRlp4b0tpdmNVS0thbjZmXzE3MjQ3MzcxNTI6MTcyNDc0MDc1Ml9WNA)