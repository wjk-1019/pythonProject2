"""
数据工具模块
用于加载、准备和处理图像数据

数据格式: data/*_frames/photos/fullbody/*.jpg + *.avi
"""
import os
from PIL import Image
from typing import List, Dict, Any

# 导入配置
from config import DATA_DIR


def prepare_data() -> List[Dict[str, str]]:
    """
    扫描 DATA_DIR 下的所有数据目录中的图片。
    
    _frames格式: data/*_frames/photos/fullbody/*.jpg
    
    Returns:
        数据列表，每个元素包含 image_path 和 text
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已创建 '{DATA_DIR}' 目录，请添加数据文件夹")
        return []

    raw_data: List[Dict[str, str]] = []

    # 遍历 data 目录下的所有子目录
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # _frames格式: photos/fullbody
        photos_root = os.path.join(folder_path, "photos", "fullbody")
        if os.path.exists(photos_root):
            for name in os.listdir(photos_root):
                lower = name.lower()
                if lower.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(photos_root, name)
                    
                    # 使用简单描述（实际应用中应从 LanceDB 读取预生成的描述）
                    text = _get_simple_description("fullbody")
                    
                    raw_data.append({
                        "image_path": image_path,
                        "text": text,
                    })

    if not raw_data:
        print(f"在 '{DATA_DIR}' 中未找到有效的图像文件")

    return raw_data


def _get_simple_description(subdir: str) -> str:
    """
    根据子目录类型生成简单的中文描述
    
    Args:
        subdir: 子目录名称 (face, ped, fullbody 等)
        
    Returns:
        中文描述文本
    """
    subdir_lower = subdir.lower()
    
    if "face" in subdir_lower:
        return "一个人的面部"
    elif "ped" in subdir_lower or "fullbody" in subdir_lower:
        return "一个行人"
    else:
        return "一个人"


def load_image(image_path: str) -> Image.Image:
    """
    加载并返回 PIL Image 对象
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL Image 对象，或 None 如果加载失败
    """
    try:
        img = Image.open(image_path)
        # 转换为 RGB 模式（确保兼容性）
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        print(f"错误: 未找到图像文件 {image_path}")
        return None
    except Exception as e:
        print(f"加载图像错误 {image_path}: {e}")
        return None


def get_all_ped_images() -> List[str]:
    """
    获取所有行人图像的路径
    
    Returns:
        行人图像路径列表
    """
    import glob
    images = []
    
    # _frames格式
    pattern = os.path.join(DATA_DIR, "*_frames", "photos", "fullbody", "*.jpg")
    images.extend(glob.glob(pattern))
    
    return sorted(images)


def get_all_face_images() -> List[str]:
    """
    获取所有人脸图像的路径
    
    Returns:
        人脸图像路径列表
    """
    import glob
    pattern = os.path.join(DATA_DIR, "*", "images", "face", "*.jpg")
    return sorted(glob.glob(pattern))


def get_all_images() -> List[str]:
    """
    获取所有图像的路径
    
    Returns:
        图像路径列表
    """
    import glob
    images = []
    
    # _frames格式: photos/fullbody
    pattern = os.path.join(DATA_DIR, "*_frames", "photos", "fullbody", "*.jpg")
    images.extend(glob.glob(pattern))
    pattern = os.path.join(DATA_DIR, "*_frames", "photos", "fullbody", "*.png")
    images.extend(glob.glob(pattern))
    
    return sorted(images)
