"""
根据cpm的输出框做简单的画框
"""

from PIL import Image, ImageDraw, ExifTags

def draw_rectangle_on_image(image_path, output_path,x1,y1,x2,y2):
    # 打开图片
    image = Image.open(image_path)
    
    # 处理EXIF旋转信息
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image doesn't have getexif
        pass

    # 获取图片宽度和高度
    width, height = image.size

    top_left = (int(x1 * width/1000), int(y1*height/1000))  # 左上角坐标
    bottom_right = (int(x2*width/1000), int(y2*height/1000))  # 右下角坐标

    # 画红色矩形框
    draw = ImageDraw.Draw(image)
    draw.rectangle([top_left, bottom_right], outline="red", width=3)

    # 保存图片
    image.save(output_path)

#找到图中每一个挖掘机的坐标。以格式(x1,y1,x2,y2)输出，其中(x1,y1)是检测框左上角坐标，(x2,y2)是右下角坐标
image_path = "./dataset/images/demo.jpg"
output_path = "/dataset/demo_box.jpg"
x1,y1,x2,y2 = 195,138,912,916
draw_rectangle_on_image(image_path, output_path, x1,y1,x2,y2)


