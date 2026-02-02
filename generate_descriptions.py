"""
文本描述生成模块
使用 Qwen2-VL 为行人图片生成详细的中文描述
独立于向量数据库重建，方便单独运行
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


def scan_all_images() -> List[str]:
    """扫描 DATA_DIR 下所有图片"""
    image_paths = []
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"数据目录不存在: {DATA_DIR}")
        return image_paths
    
    # 遍历所有子目录
    for folder in data_path.iterdir():
        if not folder.is_dir():
            continue
        
        # _frames格式: photos/fullbody
        fullbody_dir = folder / "photos" / "fullbody"
        if fullbody_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_file in fullbody_dir.glob(ext):
                    image_paths.append(str(img_file.absolute()))
    
    return sorted(image_paths)


def generate_descriptions(
    device: str = "cpu",
    max_images: Optional[int] = None,
    output_file: str = "output/image_text_pairs.json"
) -> List[dict]:
    """
    为所有图片生成文本描述
    
    Args:
        device: "cpu" 或 "cuda"
        max_images: 最大处理图片数量（None=全部）
        output_file: 输出JSON文件路径
        
    Returns:
        图文对列表
    """
    # 1. 扫描所有图片
    print("=" * 60)
    print("步骤 1: 扫描图片文件")
    print("=" * 60)
    image_paths = scan_all_images()
    
    if not image_paths:
        print("❌ 未找到任何图片文件")
        return []
    
    print(f"✓ 找到 {len(image_paths)} 张图片")
    
    if max_images:
        image_paths = image_paths[:max_images]
        print(f"  限制处理: {len(image_paths)} 张")
    
    # 2. 加载 Qwen2-VL 模型
    print("\n" + "=" * 60)
    print("步骤 2: 加载 Qwen2-VL 模型")
    print("=" * 60)
    vlm = VLMGenerator(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device=device
    )
    print(f"✓ VLM 模型加载完成，设备: {device}")
    
    # 3. 生成描述
    print("\n" + "=" * 60)
    print("步骤 3: 生成文本描述")
    print("=" * 60)
    
    pairs = []
    failed_images = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="生成描述"), 1):
        try:
            # 加载图片
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 生成详细的中文描述
            text_description = vlm.generate(
                image,
                prompt=(
                    "请详细描述这个行人的外观特征，包括："
                    "性别、年龄段、上衣颜色和类型、下装颜色和类型、"
                    "发型、配饰（如眼镜、帽子、包等）。"
                    "用一句话完整描述，不要分点列举。"
                )
            )
            
            pairs.append({
                "id": f"img_{idx:05d}",
                "image_path": image_path,
                "text": text_description
            })
            
            # 每100张显示一次进度
            if idx % 100 == 0:
                print(f"\n已处理 {idx} 张图片...")
            
        except Exception as e:
            failed_images.append((image_path, str(e)))
            print(f"\n❌ 处理失败: {Path(image_path).name} - {e}")
            continue
    
    print(f"\n✓ 成功生成: {len(pairs)} 条描述")
    if failed_images:
        print(f"✗ 失败: {len(failed_images)} 张")
    
    # 4. 保存到文件
    print("\n" + "=" * 60)
    print("步骤 4: 保存图文对数据")
    print("=" * 60)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 图文对已保存到: {output_path}")
    print(f"  共 {len(pairs)} 条记录")
    
    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成行人文本描述")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="设备选择 (cpu 或 cuda)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大处理图片数量（用于测试）")
    parser.add_argument("--output", type=str, default="output/image_text_pairs.json",
                        help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    generate_descriptions(
        device=args.device,
        max_images=args.max_images,
        output_file=args.output
    )
