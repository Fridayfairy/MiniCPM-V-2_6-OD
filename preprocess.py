import json
from PIL import Image, ImageDraw
import os
from sklearn.cluster import KMeans
import numpy as np
import random
from tqdm import tqdm
import sys

# 获取格式化的json文件，按照官方格式，仅在输出中增加了<box>x1,y1,x2,y2</box>作为
def get_query_answer(boxes, textes):
    assert len(boxes) == len(textes)
    # 这里可以根据任务需要去修改你的query question
    query_list = [
        "请识别图像内的挖掘机，并附上它们的坐标。",
        "我想知道图中挖掘机的位置。",
        "Please help me identify the construction machinery in the picture, and mark their exact locations.",
        "I need you to find all the construction machinery and give their location information.",
    ]
    query = random.choice(query_list)
    format_answer = ""
    # 对坐标进行归一化
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        format_answer += "<ref>{text}<box>{x1}</box><box>{y1}</box><box>{x2}</box><box>{y2}</box></ref>".format(
            text=textes[index], x1=x1, y1=y1, x2=x2, y2=y2
        )
    return query, format_answer

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_to_json(output_data, output_path):
    random.shuffle(output_data)
    print('共有数据：',len(output_data))
    len_val = int(0.1 * len(output_data))
    with open(
        os.path.join(output_path,"train_set.json"), "w"
    ) as file:
        # json.dump(output_data[:-len_val], file, ensure_ascii=False, indent=4)
        json.dump(output_data, file, ensure_ascii=False, indent=4) # demo里仅提供了一张图片
    with open(
       os.path.join(output_path,"test_set.json"), "w"
    ) as file:
        # json.dump(output_data[-len_val:], file, ensure_ascii=False, indent=4)
        json.dump(output_data, file, ensure_ascii=False, indent=4) # demo里仅提供了一张图片
    return
 
def main(max_dataset=sys.maxsize):
    data=load_json(json_path)
    all_data_dict = data["data"]
    output_data = []
    index = 0
    for k, v in tqdm(all_data_dict.items()):
        out_dict = {}
        gt = v["gt"]
        boxes = [box["bbox"] for box in gt]
        textes = [box["text"] for box in gt]

        out_dict["id"] = str(index)
        out_dict["image"] = k

        query, format_answer = get_query_answer(boxes, textes)
        out_dict["conversations"] = [
            {"content": "<image>\n{}".format(query), "role": "user"},
            {"content": format_answer, "role": "assistant"},
        ]
        output_data.append(out_dict)
        index += 1
        if index >= max_dataset:
            break
    save_to_json(output_data, output_path)

if __name__ == "__main__":
    json_path="./dataset/dataset.json"
    output_path='./dataset'
    main()
