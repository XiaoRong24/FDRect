from PIL import Image, ImageFilter
import os
import numpy as np


def process_images_no_scipy(image_folder, mask_folder, output_folder, threshold=30):
    """
    不使用scipy的简化版本
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file)

        if not os.path.exists(mask_path):
            print(f"警告: 找不到对应的掩码文件 {mask_path}")
            continue

        try:
            # 打开图像和掩码
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            if image.size != mask.size:
                mask = mask.resize(image.size)

            image_array = np.array(image)
            mask_array = np.array(mask)

            # 创建白色背景
            output_array = np.full_like(image_array, 255)

            # 使用较低的阈值来包含更多边缘像素
            content_mask = mask_array > threshold

            # 复制内容区域
            output_array[content_mask] = image_array[content_mask]

            # 对最终图像进行轻微模糊来平滑边缘
            output_image = Image.fromarray(output_array)
            output_image = output_image.filter(ImageFilter.GaussianBlur(radius=0.5))

            output_path = os.path.join(output_folder, image_file)
            output_image.save(output_path, quality=100)

            print(f"已处理: {image_file}")

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")


# 使用这个版本
if __name__ == "__main__":
    image_folder = "./TrueStitching/testing/input"
    mask_folder = "./TrueStitching/testing/mask"
    output_folder = "./TrueStitching/testing/output_clean"

    process_images_no_scipy(image_folder, mask_folder, output_folder, threshold=30)
    print("处理完成！")