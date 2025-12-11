import json
###将训练集标签的txt文件转换成json文件
def txt_to_json(txt_file_path, json_file_path):
    data = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            i=0
            for line in txt_file:
                # 去除换行符
                line = line.strip()
                if line:
                    # 按制表符分割每行内容
                    parts = line.split('\t')
                    if len(parts) == 2:
                        image_name, risk_level = parts
                        image_path = '/data1/jianghui/eleme_2/data/train/'+risk_level+'/'+image_name
                        data.append( {
                    "id": f"identity_{i+1}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": f"请判断这张图片属于以下哪一类：非楼道，无风险，低风险，中风险，高风险: <|vision_start|>{image_path}<|vision_end|>"
                        },
                        {
                            "from": "assistant", 
                            "value": risk_level
                        }
                    ]
                }
                        
                        )
                        i=i+1
        # 将数据写入 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"已成功将 {txt_file_path} 转换为 {json_file_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {txt_file_path}")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    txt_file_path = 'train_process.txt'
    json_file_path = 'data_vl.json'
    txt_to_json(txt_file_path, json_file_path)    