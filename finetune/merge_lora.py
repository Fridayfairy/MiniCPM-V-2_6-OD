"""
# Author： YCX
# Date:    2024.08.27
# Description:  把生成的lora文件合并到源模型中
"""

from peft import PeftModel
from transformers import AutoModel,AutoTokenizer
import shutil
import os
model_type="/data1/home/ycx/workspace/HF/MiniCPM-V-2_6" # 官方下载的原始模型
path_to_adapter="/data1/home/ycx/ycxGit/MiniCPM-V26/finetune/output/output__lora/checkpoint-10000" # lora保存的地址
merge_path="./MiniCPM-V-2_6-OD" # 希望将lora合并到主模型后的保存地址

model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
        )
# 挂载lora模块
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()
print("load lora ok")
# 合并lora模块到原模型，模型shape与原始MiniCPM-Llama3-V-2_5相同
merge_model=lora_model.merge_and_unload()
# 保存新的模型，与原始MiniCPM-Llama3-V-2_5的shape相同
merge_model.save_pretrained(merge_path,safe_serialization=False)
print(f"save {merge_path}")

# 加载分词文件与模型保存到merge后的模型地址
tokenizer=AutoTokenizer.from_pretrained(model_type,trust_remote_code=True)
tokenizer.save_pretrained(merge_path)

# 将缺少的文件复制到新模型路径下
copy_files = ['image_processing_minicpmv.py', 'preprocessor_config.json', 'processing_minicpmv.py']

for file_name in copy_files:
    src_path = os.path.join(model_type, file_name)
    dst_path = os.path.join(merge_path, file_name)

    # 复制文件
    try:
        shutil.copy(src_path, dst_path)
        print(f"File '{src_path}' has been copied to '{dst_path}'")
    except shutil.Error as e:
        # 处理文件复制中的错误，例如文件不存在或权限问题
        print(f"Error: {e}")
    except IOError as e:
        # 处理文件操作中的I/O错误
        print(f"I/O Error: {e}")

