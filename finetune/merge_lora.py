"""

https://modelbest.feishu.cn/wiki/MPkPwvONEiZm3BkWMnyc83Tin4d
"""
from peft import PeftModel
from transformers import AutoModel,AutoTokenizer
model_type="/data1/home/ycx/workspace/HF/MiniCPM-V-2_6" # local_model_path or openbmb/MiniCPM-V-2.5
path_to_adapter="/data1/home/ycx/ycxGit/MiniCPM-V26/finetune/output/output__lora/checkpoint-10000" # lora保存的地址
merge_path="./merged_cpm26" # 希望将lora合并到主模型后的保存地址

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

