import os
import shutil

"""
将train里的图片按类存放
"""
def move_images_by_category(image_folder, txt_file):
    # 读取 txt 文件中的图片名和类别信息
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 存储每个类别的图片列表
    category_images = {}
    for line in lines:
        image_name, category = line.strip().split('\t')
        if category not in category_images:
            category_images[category] = []
        category_images[category].append(image_name)

    # 创建类别文件夹并移动图片
    for category, images in category_images.items():
        category_folder = os.path.join(os.getcwd(), category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        for image in images:
            image_path = os.path.join(image_folder, image)
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(category_folder, image))
                print(f"Moved {image} to {category} folder.")
            else:
                print(f"Image {image} not found in the image folder.")
    print('全部移动完毕已经！')

if __name__ == "__main__":
    # 请替换为实际的图片文件夹路径和 txt 文件路径
    image_folder = './train/'
    txt_file = './label/train.txt'
    move_images_by_category(image_folder, txt_file)
    # 要移动的五个文件夹的路径列表
    folder_paths = [
        './低风险',
        './非楼道',
        './高风险',
        './无风险',
        './中风险'
    ]

    # 新创建的 data 文件夹路径
    data_folder = './data/train'

    # 创建 data 文件夹
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # 遍历每个文件夹路径并移动到 data 文件夹下
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        destination_folder = os.path.join(data_folder, folder_name)
        shutil.move(folder_path, destination_folder)

    print("文件夹移动完成！")

        