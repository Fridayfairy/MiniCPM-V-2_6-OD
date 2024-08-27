"""
# Author： YCX
# Date:    2024.08.22
# Description:  把labelimg标注的xml格式框，修改为MiniCPM所需格式
"""

import json
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def xml_to_json(xml_path, json_dict):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 获取图片的绝对路径
    path_tmp = os.path.abspath(xml_path).replace('/Annotations/', '/images/')
    
    for ext in ['.jpg', '.JPG', '.png']:
        image_path = os.path.abspath(path_tmp).replace('.xml', ext)
        if os.path.exists(image_path):
            break
    else:
        # print(f"no {xml_path}")
        return

    # 初始化图片的标注列表
    gt_list = []

    # 遍历所有的object元素
    for obj in root.findall('object'):
        # 创建一个新的标注字典
        gt_dict = {

        }
        # 获取文本内容
        class_name = obj.find('name').text
        if class_name in name_2_text.keys():
                class_name = name_2_text[class_name]
        else:
            continue
        gt_dict["text"] = class_name

        # 获取bounding box坐标
        bndbox = obj.find('bndbox')
        x1 = float(bndbox.find('xmin').text)
        y1 = float(bndbox.find('ymin').text)
        x2 = float(bndbox.find('xmax').text)
        y2 = float(bndbox.find('ymax').text)

        # 2. 将所有定位框坐标按照图像的高和宽归一化到(0,1000).即x=x*1000/width，y=y*1000/height  merge_box中实现
        x1 = int(x1 * 1000 / width)
        x2 = int(x2 * 1000 / width)
        y1 = int(y1 * 1000 / height)
        y2 = int(y2 * 1000 / height)
        
        # 添加四个顶点坐标
        gt_dict["bbox"] = [x1, y1, x2, y2]

        # 将标注添加到列表中
        gt_list.append(gt_dict)

    # 将图片路径和标注列表添加到字典中
    if gt_list:
        json_dict[image_path] = {
            "gt": gt_list,
            "image_path": image_path
        }

name_2_text = {
    'wajueji': '挖掘机',
}

if __name__ == "__main__":
    xml_directory = './dataset/Annotations/'  # 替换为你的XML文件所在目录
    xml_files = []
    for xml_file in tqdm(os.listdir(xml_directory)):
        if xml_file.endswith('.xml'):
            xml_files.append(os.path.join(xml_directory, xml_file))
    print(f"find {len(xml_files)} xml")

    # 初始化JSON字典
    json_output = {"data": {}}

    # 遍历所有的XML文件并转换
    for xml_file in tqdm(xml_files):
        xml_to_json(xml_file, json_output["data"])

    # 将字典转换为JSON字符串并保存到文件
    with open('./dataset/dataset.json', 'w') as json_file:
        json.dump(json_output, json_file, ensure_ascii=False, indent=4)

    print("JSON转换完成并保存。")



