"""
改进版文本描述生成器
使用更详细的提示词生成高区分度的行人描述
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from typing import List, Optional

from text_generation.vlm_generator import VLMGenerator
from config import DATA_DIR


# 改进的提示词模板 - 要求更具体的描述
IMPROVED_PROMPT = """仔细观察这张行人图片，生成一段详细且独特的外观描述。

要求：
1. 颜色描述要具体（如"深蓝色"、"酒红色"、"米白色"，避免使用"深色"、"浅色"）
2. 服装类型要准确（如"格子衬衫"、"运动连帽衫"、"牛仔夹克"）
3. 描述独特细节（如衣服上的图案、logo、条纹、口袋位置等）
4. 包含姿态或动作（如"低头看手机"、"双手插兜"、"小跑步行"）
5. 如果可见，描述发型特征（如"黑色短发"、"棕色马尾辫"、"秃顶"）
6. 描述配饰细节（眼镜框架颜色、帽子款式、包的类型和颜色）

用一段流畅的中文描述这个人，确保描述足够详细和独特，能够将此人与其他行人区分开。不要使用"可能"、"似乎"等不确定词汇。"""


def scan_all_images() -> List[str]:
    """扫描 DATA_DIR 下所有图片"""
    image_paths = []
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"数据目录不存在: {DATA_DIR}")
        return image_paths
    
    for folder in data_path.iterdir():
        if not folder.is_dir():
            continue
        
        fullbody_dir = folder / "photos" / "fullbody"
        if fullbody_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_file in fullbody_dir.glob(ext):
                    image_paths.append(str(img_file.absolute()))
    
    return sorted(image_paths)


def generate_improved_descriptions(
    device: str = "cuda",
    max_images: Optional[int] = None,
    output_file: str = "output/improved_pairs.json"
) -> List[dict]:
    """
    使用改进的提示词生成更高质量的文本描述
    
    Args:
        device: "cpu" 或 "cuda"
        max_images: 最大处理图片数量
        output_file: 输出文件路径
    """
    print("=" * 60)
    print("改进版文本描述生成")
    print("=" * 60)
    
    # 扫描图片
    image_paths = scan_all_images()
    if not image_paths:
        print("❌ 未找到任何图片")
        return []
    
    print(f"✓ 找到 {len(image_paths)} 张图片")
    
    if max_images:
        image_paths = image_paths[:max_images]
        print(f"  限制处理: {len(image_paths)} 张")
    
    # 加载模型
    print("\n加载 Qwen2-VL 模型...")
    vlm = VLMGenerator(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device=device
    )
    print(f"✓ 模型加载完成")
    
    # 生成描述
    print("\n开始生成改进的描述...")
    pairs = []
    failed = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="生成描述"), 1):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 使用改进的提示词
            description = vlm.generate(image, prompt=IMPROVED_PROMPT)
            
            # 基本清理
            description = description.strip()
            if len(description) < 20:
                # 描述太短，尝试再次生成
                description = vlm.generate(image, prompt=IMPROVED_PROMPT)
            
            pairs.append({
                "id": f"img_{idx:05d}",
                "image_path": image_path,
                "text": description
            })
            
        except Exception as e:
            failed.append((image_path, str(e)))
            print(f"\n❌ 失败: {Path(image_path).name} - {e}")
    
    print(f"\n✓ 成功: {len(pairs)} 条")
    if failed:
        print(f"✗ 失败: {len(failed)} 条")
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 保存到: {output_path}")
    
    # 显示样例
    print("\n" + "=" * 60)
    print("生成样例（前5条）")
    print("=" * 60)
    for p in pairs[:5]:
        print(f"\n📷 {Path(p['image_path']).name}")
        print(f"   {p['text']}")
    
    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="改进版文本描述生成")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output", type=str, default="output/improved_pairs.json")
    
    args = parser.parse_args()
    
    generate_improved_descriptions(
        device=args.device,
        max_images=args.max_images,
        output_file=args.output
    )
