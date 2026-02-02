"""
向量数据库重建模块
使用 Chinese-CLIP (cn_clip) 为已有的图文对生成向量并存入 LanceDB
独立于文本描述生成，可单独运行
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from typing import List, Optional

from utils import lancedb_utils, vector_utils
from config import DATA_DIR


def rebuild_vector_database(
    input_file: str = "output/image_text_pairs.json",
    batch_size: int = 50
):
    """
    从已有的图文对JSON文件重建向量数据库
    
    Args:
        input_file: 图文对JSON文件路径
        batch_size: 批处理大小
    """
    # 1. 读取图文对数据
    print("=" * 60)
    print("步骤 1: 读取图文对数据")
    print("=" * 60)
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_file}")
        print("请先运行 generate_descriptions.py 生成图文对数据")
        return
    
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"✓ 读取 {len(pairs)} 条图文对")
    
    # 2. 加载 CLIP 模型
    print("\n" + "=" * 60)
    print("步骤 2: 加载 Chinese-CLIP 模型")
    print("=" * 60)
    
    # 触发模型加载
    vector_utils.load_clip_model()
    vector_dim = vector_utils.get_vector_dimension()
    print(f"✓ 向量维度: {vector_dim}")
    
    # 3. 批量计算向量
    print("\n" + "=" * 60)
    print("步骤 3: 计算图像和文本向量")
    print("=" * 60)
    
    records = []
    failed_items = []
    
    for i in tqdm(range(0, len(pairs), batch_size), desc="处理批次"):
        batch = pairs[i:i+batch_size]
        
        # 加载批次图片
        images = []
        texts = []
        valid_indices = []
        
        for j, pair in enumerate(batch):
            try:
                img = Image.open(pair["image_path"])
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)
                texts.append(pair["text"])
                valid_indices.append(j)
            except Exception as e:
                failed_items.append((pair["image_path"], str(e)))
        
        if not images:
            continue
        
        # 批量计算向量
        image_vectors = vector_utils.get_batch_image_vectors(images)
        text_vectors = vector_utils.get_batch_text_vectors(texts)
        
        # 组装记录
        for k, j in enumerate(valid_indices):
            if image_vectors[k] is not None and text_vectors[k] is not None:
                records.append({
                    "image_path": batch[j]["image_path"],
                    "text": batch[j]["text"],
                    "vector_image": image_vectors[k],
                    "vector_text": text_vectors[k]
                })
            else:
                failed_items.append((batch[j]["image_path"], "向量计算失败"))
    
    print(f"\n✓ 成功处理: {len(records)} 条")
    if failed_items:
        print(f"✗ 失败: {len(failed_items)} 条")
    
    if not records:
        print("❌ 没有成功处理任何数据")
        return
    
    # 4. 重建向量数据库
    print("\n" + "=" * 60)
    print("步骤 4: 重建向量数据库")
    print("=" * 60)
    
    db = lancedb_utils.initialize_lancedb()
    
    # 删除旧表
    try:
        db.drop_table("multimodal_data")
        print("✓ 已删除旧数据表")
    except:
        print("  没有旧数据表需要删除")
    
    # 创建新表
    table = lancedb_utils.get_or_create_table(db, vector_dim)
    
    # 批量添加数据
    db_batch_size = 100
    for i in range(0, len(records), db_batch_size):
        batch = records[i:i+db_batch_size]
        lancedb_utils.add_data_to_table(table, batch)
        print(f"  添加批次 {i//db_batch_size + 1}: {len(batch)} 条")
    
    # 5. 验证数据库
    print("\n" + "=" * 60)
    print("步骤 5: 验证数据库")
    print("=" * 60)
    
    count = lancedb_utils.get_table_stats(table)
    
    # 显示示例记录
    print("\n前 3 条记录示例:")
    sample = table.to_pandas().head(3)
    for idx, row in sample.iterrows():
        print(f"\n{idx + 1}. {Path(row['image_path']).name}")
        print(f"   描述: {row['text'][:50]}...")
        print(f"   图片向量维度: {len(row['vector_image'])}")
        print(f"   文本向量维度: {len(row['vector_text'])}")
    
    # 6. 完成总结
    print("\n" + "=" * 60)
    print("✅ 数据库重建完成！")
    print("=" * 60)
    print(f"总计处理: {len(records)} 条记录")
    print(f"数据库路径: {db.uri}")
    print(f"向量维度: {vector_dim}")


def rebuild_from_existing_pairs():
    """
    从已有的 regenerated_pairs.json 重建向量数据库
    这是一个便捷函数，使用之前生成的图文对
    """
    existing_file = "output/regenerated_pairs.json"
    if Path(existing_file).exists():
        rebuild_vector_database(input_file=existing_file)
    else:
        print(f"❌ 文件不存在: {existing_file}")
        print("请先运行 generate_descriptions.py 或使用已有数据")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="重建向量数据库")
    parser.add_argument("--input", type=str, default="output/regenerated_pairs.json",
                        help="图文对JSON文件路径")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="批处理大小")
    
    args = parser.parse_args()
    
    rebuild_vector_database(
        input_file=args.input,
        batch_size=args.batch_size
    )
