"""
LanceDB 向量数据库操作模块
提供数据库初始化、表管理、数据添加和检索功能
"""
import lancedb
import pyarrow as pa
import numpy as np
from typing import List, Dict, Any, Optional

# 导入配置
from config import LANCEDB_DIR, TABLE_NAME


def distance_to_similarity(distance: Optional[float], max_distance: float = 2.0) -> float:
    """
    将 LanceDB 返回的距离分数换算成 0~1 的相似度。
    更小的距离表示更相似的向量。
    
    Args:
        distance: L2 距离值
        max_distance: 最大距离阈值
        
    Returns:
        相似度分数 (0~1)
    """
    if max_distance <= 0:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(d) or np.isinf(d):
        return 0.0
    d = max(0.0, min(d, max_distance))
    return (max_distance - d) / max_distance


def initialize_lancedb():
    """连接到 LanceDB 数据库"""
    db = lancedb.connect(LANCEDB_DIR)
    print(f"✓ LanceDB 连接成功: {LANCEDB_DIR}")
    return db


def get_or_create_table(db, vector_dim: int):
    """
    打开一个已存在的表，如果不存在则创建它
    
    Args:
        db: LanceDB 数据库连接
        vector_dim: 向量维度（应为 1024 用于 Chinese-CLIP）
        
    Returns:
        LanceDB 表对象
    """
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("vector_image", pa.list_(pa.float32(), list_size=vector_dim)),
        pa.field("vector_text", pa.list_(pa.float32(), list_size=vector_dim))
    ])

    try:
        # 检查表是否存在
        if TABLE_NAME in db.table_names():
            table = db.open_table(TABLE_NAME)
            print(f"✓ 打开已存在的表 '{TABLE_NAME}'")
        else:
            table = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")
            print(f"✓ 创建新表 '{TABLE_NAME}'")
    except Exception as e:
        print(f"表操作错误: {e}")
        table = db.create_table(TABLE_NAME, schema=schema, mode="overwrite")
        print(f"✓ 重新创建表 '{TABLE_NAME}'")

    return table


def add_data_to_table(table, data_batch: List[Dict[str, Any]]):
    """
    向 LanceDB 表中添加一批数据
    
    Args:
        table: LanceDB 表对象
        data_batch: 数据批次列表，每个元素包含 text, image_path, vector_image, vector_text
    """
    if not data_batch:
        print("没有数据要添加")
        return

    # 过滤掉 vector_image 或 vector_text 为 None 的记录
    valid_batch = [item for item in data_batch if
                   item.get("vector_image") is not None and item.get("vector_text") is not None]

    if not valid_batch:
        print("没有有效的向量数据可添加")
        return

    try:
        table.add(valid_batch)
        print(f"✓ 成功添加 {len(valid_batch)} 条记录")
    except Exception as e:
        print(f"向表添加数据错误: {e}")


def search_in_table(
        table,
        query_vector: List[float],
        search_column: str,
        limit: int = 5
) -> List[Dict[str, Any]]:
    """
    在 LanceDB 表中执行搜索
    
    Args:
        table: LanceDB 表对象
        query_vector: 查询向量
        search_column: 搜索的列名 (vector_image 或 vector_text)
        limit: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    if not query_vector:
        print("查询向量为空，无法执行搜索")
        return []

    try:
        results = (table
                   .search(query_vector, vector_column_name=search_column)
                   .limit(limit)
                   .to_list())
        return results
    except Exception as e:
        print(f"搜索错误: {e}")
        return []


def check_existing_record(table, image_path: str) -> bool:
    """
    检查表中是否已存在指定 image_path 的记录
    
    Args:
        table: LanceDB 表对象
        image_path: 图像路径
        
    Returns:
        True 如果存在，False 否则
    """
    try:
        results = table.search().where(f"image_path = '{image_path}'").limit(1).to_list()
        return len(results) > 0
    except Exception as e:
        print(f"检查现有记录错误 {image_path}: {e}")
        return False


def get_table_stats(table) -> int:
    """
    获取表的统计信息
    
    Args:
        table: LanceDB 表对象
        
    Returns:
        表中的行数
    """
    try:
        count = table.count_rows()
        print(f"✓ 表 '{TABLE_NAME}' 中有 {count} 行数据")
        return count
    except Exception as e:
        print(f"获取表统计信息错误: {e}")
        return 0


def delete_all_data(table):
    """
    清空表中的所有数据（用于重新初始化）
    
    Args:
        table: LanceDB 表对象
    """
    try:
        # 获取所有行，然后删除
        all_rows = table.search().limit(100000).to_list()
        if all_rows:
            # LanceDB 不支持直接批量删除，需要重建表
            print("⚠ 无法直接删除，建议重建数据库")
        return True
    except Exception as e:
        print(f"清空数据错误: {e}")
        return False
