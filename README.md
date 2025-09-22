# eleme_ai_wise_knight
**项目简介**

ELE AI算法大赛赛道二：智慧骑士—消防隐患识别比赛的参考代码，数据集来自官方数据集，整体采用多模态大模型微调的思路，最终得分B榜4.9左右，A榜5.5左右。

---
## 背景

对“消防隐患随手拍”项目中的拍摄照片内容进行识别，实时判断照片内场景是否存在消防安全隐患以及隐患的危险程度。根据楼道中是否存在大量堆积物，堆积物是否可燃以及是否有起火风险将隐患分为无隐患、低风险、中等风险、高风险。比赛赛题介绍：(https://tianchi.aliyun.com/competition/entrance/532324)
数据集下载：(https://tianchi.aliyun.com/competition/entrance/532324/information)

---
## 技术栈
- **大模型**：Qwen2.5-VL-7B-Instruct
- **语言**：Python
- **监控工具**：SwanLab
---
## 安装与运行
**环境要求**

半精度训练一张RTX 3090足够，全精度需要两张RTX 3090。

**运行步骤**

1. 数据集处理(将训练集里的图片按类存放)：
  ```bash
  python data_process.py 
  ```
2. 下载多模态大模型：
  ```bash
  python download_qwen.py 
  ```
3. 训练模型：  
  ```bash
  python qwen25_llm_lora_train_v6.py 
  ```
4. 模型推理和预测：
  ```bash
  python qwen25_llm_lora_infer_v6.py 
  ```
5.  训练过程：
![alt text](image.png)
    
6.  预测结果：
![alt text](image-1.png)
![alt text](image-2.png)


