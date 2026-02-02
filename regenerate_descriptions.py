"""
使用 Qwen2-VL 重新生成高质量的图文对并完全重建向量数据库

特点：
1. 为所有图片生成详细的中文描述（使用 Qwen2-VL）
2. 重新计算所有图片向量和文本向量（使用 Chinese-CLIP）
3. 确保图文对齐质量，提高检索相似度
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

from text_generation.vlm_generator import VLMGenerator
from utils import lancedb_utils, vector_utils
from config import DATA_DIR

def scan_all_images():
    """扫描 DATA_DIR 下所有图片（支持两种数据格式）"""
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

def regenerate_all_vectors(device="cpu", max_images=None):
    """
    完全重建向量数据库
    
    Args:
        device: "cpu" 或 "cuda"
        max_images: 最大处理图片数量（None=全部）
    """
    
    # 1. 扫描所有图片
    print("=" * 60)
    print("步骤 1: 扫描图片文件")
    print("=" * 60)
    image_paths = scan_all_images()
    
    if not image_paths:
        print("❌ 未找到任何图片文件")
        return
    
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
    
    # 3. 生成描述并计算向量
    print("\n" + "=" * 60)
    print("步骤 3: 生成描述并计算向量")
    print("=" * 60)
    
    records = []
    failed_images = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="处理图片"), 1):
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
            
            # 计算图片向量
            image_vector = vector_utils.get_image_vector(image)
            
            # 计算文本向量
            text_vector = vector_utils.get_text_vector(text_description)
            
            if image_vector is None or text_vector is None:
                failed_images.append((image_path, "向量计算失败"))
                continue
            
            # 添加到记录
            records.append({
                "image_path": image_path,
                "text": text_description,
                "vector_image": image_vector,
                "vector_text": text_vector
            })
            
            # 每10张显示一次示例
            if idx % 10 == 0 and idx <= 30:
                print(f"\n示例 {idx}: {Path(image_path).name}")
                print(f"  描述: {text_description}")
            
        except Exception as e:
            failed_images.append((image_path, str(e)))
            print(f"\n❌ 处理失败: {Path(image_path).name} - {e}")
            continue
    
    print(f"\n✓ 成功处理: {len(records)} 张")
    if failed_images:
        print(f"✗ 失败: {len(failed_images)} 张")
    
    if not records:
        print("❌ 没有成功处理任何图片")
        return
    
    # 4. 保存图文对到 JSON（备份）
    print("\n" + "=" * 60)
    print("步骤 4: 保存图文对数据")
    print("=" * 60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    pairs_file = output_dir / "regenerated_pairs.json"
    pairs_data = [
        {
            "id": f"img_{i:05d}",
            "image_path": r["image_path"],
            "text": r["text"]
        }
        for i, r in enumerate(records)
    ]
    
    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(pairs_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 图文对已保存到: {pairs_file}")
    
    # 5. 重建向量数据库
    print("\n" + "=" * 60)
    print("步骤 5: 重建向量数据库")
    print("=" * 60)
    
    db = lancedb_utils.initialize_lancedb()
    vector_dim = vector_utils.get_vector_dimension()
    
    # 删除旧表
    try:
        db.drop_table("multimodal_data")
        print("✓ 已删除旧数据表")
    except:
        print("  没有旧数据表需要删除")
    
    # 创建新表
    table = lancedb_utils.get_or_create_table(db, vector_dim)
    
    # 批量添加数据
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        lancedb_utils.add_data_to_table(table, batch)
        print(f"  添加批次 {i//batch_size + 1}: {len(batch)} 条")
    
    # 6. 验证数据库
    print("\n" + "=" * 60)
    print("步骤 6: 验证数据库")
    print("=" * 60)
    
    count = lancedb_utils.get_table_stats(table)
    
    # 显示示例记录
    print("\n前 3 条记录示例:")
    import pandas as pd
    sample = table.to_pandas().head(3)
    for idx, row in sample.iterrows():
        print(f"\n{idx + 1}. {Path(row['image_path']).name}")
        print(f"   描述: {row['text']}")
        print(f"   图片向量维度: {len(row['vector_image'])}")
        print(f"   文本向量维度: {len(row['vector_text'])}")
    
    # 7. 完成总结
    print("\n" + "=" * 60)
    print("✅ 数据库重建完成！")
    print("=" * 60)
    print(f"总计处理: {len(records)} 条记录")
    print(f"数据库路径: {db.uri}")
    print(f"表名: multimodal_data")
    print(f"向量维度: {vector_dim}")
    print("\n现在可以启动 API 服务进行测试：")
    print("  python api_server.py")
    
    if failed_images:
        print(f"\n⚠ 以下 {len(failed_images)} 张图片处理失败:")
        for img_path, error in failed_images[:10]:
            print(f"  - {Path(img_path).name}: {error}")
        if len(failed_images) > 10:
            print(f"  ... 还有 {len(failed_images) - 10} 张")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="重建高质量向量数据库")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="设备选择 (cpu 或 cuda)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最大处理图片数量（用于测试）")
    
    args = parser.parse_args()
    
    regenerate_all_vectors(device=args.device, max_images=args.max_images)
