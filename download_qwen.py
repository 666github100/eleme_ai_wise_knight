from modelscope.hub.snapshot_download import snapshot_download

# 定义模型名称和本地保存目录
model_name = 'qwen/Qwen2.5-VL-7B-Instruct'
save_directory = './qwen_vl_7b_local'

try:
    # 下载模型和分词器到本地
    snapshot_download(model_name, cache_dir=save_directory)
    print(f"模型和分词器已成功保存到 {save_directory}")
except Exception as e:
    print(f"下载过程中出现错误: {e}")