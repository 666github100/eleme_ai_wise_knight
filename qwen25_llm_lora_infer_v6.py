import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json

"""
Qwen2.5-vl-7B-Instruct大模型微调后的推理版本,测试用
"""


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    

    return output_text[0]


model_path='./qwen_vl_7b_local/qwen/Qwen2___5-VL-7B-Instruct'
# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法


# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 推理模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen25-VL-7B-Instruct-train_v6/checkpoint-2550", config=val_config)

# 读取txt文件里的所有图片并存到列表里
def read_image_names(file_path):
    try:
        with open(file_path, 'r') as file:
            # 逐行读取文件内容，并去除每行末尾的换行符
            image_names = [line.strip() for line in file.readlines()]
        return image_names
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return []


swanlab.init(project="Qwen2.5-VL-finetune",
    experiment_name="qwen2.5-vl-test-v6-2550")

# 读取测试数据
test_dataset = read_image_names('./test/label_B/B_new.txt')

test_image_list = []
result_file = "./results_test_v6_4.txt"
for item in test_dataset:

    origin_image_path ='./test/B/B/' + item
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": origin_image_path,
            "resized_height": 560,
            "resized_width": 560,
            },
            {
            "type": "text",
            "text": "这是一个消防隐患识别分类问题，请判断这张图片属于以下哪一类：非楼道，无风险，低风险，中风险，高风险（分类规则为：高风险：楼道中出现电动车、电瓶、飞线充电等可能起火的元素；中风险：楼道中存在大量堆积物严重影响通行或堆放大量纸箱、木质家具等能造成火势蔓延的堵塞物；低风险：存在楼道堆物现象但不严重；无风险：楼道干净，无堆放物品；非楼道：一些与楼道无关的图片）"
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))
    with open(result_file, 'a') as file:
        # 组合成指定格式
        line = f"{item}\t{response}\n"
        # 写入文件
        file.write(line)

print('所有结果处理完毕！')
swanlab.log({"Prediction": test_image_list})
