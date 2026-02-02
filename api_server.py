import os
# 设置环境变量，解决 OpenMP 库冲突问题（必须在导入其他库之前设置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import mimetypes
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
from typing import Optional, Tuple, Dict
import re

# 导入新的模块结构
from utils import vector_utils, lancedb_utils
from utils.data_utils import load_image
from config import DATA_DIR

# 相似度相关常量
# Chinese-CLIP 的余弦距离范围通常在 0~2 之间
# 距离 0 表示完全相同，距离 2 表示完全相反
MAX_DISTANCE = 2.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vector_utils.load_clip_model()
vector_dim = vector_utils.get_vector_dimension()
db = lancedb_utils.initialize_lancedb()
table = lancedb_utils.get_or_create_table(db, vector_dim)

app = FastAPI(title="多模态检索系统API")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_BASE_URL = "http://localhost:8000"


def build_image_url(request: Request, image_path: str) -> str:
    if not image_path:
        return ""

    # Convert absolute path to relative path from DATA_DIR
    # 使用 normcase 处理 Windows 路径大小写问题
    abs_data_dir = os.path.abspath(DATA_DIR)
    abs_image_path = os.path.abspath(image_path)

    # Windows 路径比较需要统一大小写
    if os.path.normcase(abs_image_path).startswith(os.path.normcase(abs_data_dir)):
        rel_path = os.path.relpath(abs_image_path, abs_data_dir)
        # Ensure forward slashes for URL
        rel_path = rel_path.replace("\\", "/")
        return f"{API_BASE_URL}/data/{rel_path}"

    return ""


def build_video_url(request: Request, video_path: str, use_stream: bool = True) -> str:
    """
    构建视频URL
    
    Args:
        request: HTTP请求对象
        video_path: 视频文件的绝对路径
        use_stream: 是否使用视频流端点（推荐True，支持Range请求）
        
    Returns:
        可访问的视频URL
    """
    if not video_path:
        return ""

    abs_data_dir = os.path.abspath(DATA_DIR)
    abs_video_path = os.path.abspath(video_path)

    # Windows 路径比较需要统一大小写
    if os.path.normcase(abs_video_path).startswith(os.path.normcase(abs_data_dir)):
        rel_path = os.path.relpath(abs_video_path, abs_data_dir)
        rel_path = rel_path.replace("\\", "/")
        # 使用视频流端点（更好的兼容性和Range请求支持）
        if use_stream:
            return f"{API_BASE_URL}/video/{rel_path}"
        else:
            return f"{API_BASE_URL}/data/{rel_path}"

    return ""


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "success": False,
        "error": {
            "code": exc.status_code,
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        }
    })


def _to_percent(dist) -> float:
    """
    将 LanceDB 返回的距离分数映射为 0~100% 的相似度。
    
    针对 Chinese-CLIP 跨模态检索优化：
    - 文本-图像跨模态检索的距离通常在 1.0-1.5 范围
    - 图像-图像同模态检索的距离通常在 0.3-0.8 范围
    
    映射策略（针对跨模态优化，更宽松）：
    - 距离 0.0 → 100%
    - 距离 0.8 → 92%
    - 距离 1.2 → 82%  (典型的好匹配)
    - 距离 1.5 → 72%
    - 距离 2.0 → 55%
    """
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return 0.0
    
    # 限制距离范围 [0, 2]
    d = max(0.0, min(d, MAX_DISTANCE))
    
    # 使用分段线性映射，对跨模态检索更友好
    if d <= 0.8:
        # 非常相似（同模态或完美匹配）: 0.0->100%, 0.8->92%
        percent = 100.0 - d * 10.0
    elif d <= 1.2:
        # 相似（好的跨模态匹配）: 0.8->92%, 1.2->82%
        percent = 92.0 - (d - 0.8) * 25.0
    elif d <= 1.5:
        # 中等相似（典型跨模态匹配）: 1.2->82%, 1.5->72%
        percent = 82.0 - (d - 1.2) * 33.33
    else:
        # 弱相关: 1.5->72%, 2.0->55%
        percent = 72.0 - (d - 1.5) * 34.0
    
    return round(max(0.0, min(100.0, percent)), 2)


def _to_percent_relative(dist: float, min_dist: float, max_dist: float) -> float:
    """
    基于相对排名和绝对距离的混合方法计算相似度百分比。
    
    针对 Chinese-CLIP 跨模态检索优化：
    - 跨模态检索距离通常在 1.0-1.5，这是正常范围
    - 同模态检索距离通常在 0.3-0.8
    
    Args:
        dist: 当前结果的距离值
        min_dist: 本次搜索结果中的最小距离（最相似）
        max_dist: 本次搜索结果中的最大距离（最不相似）
    
    Returns:
        相似度百分比 (0~100)
    """
    try:
        d = float(dist)
        min_d = float(min_dist)
        max_d = float(max_dist)
    except (TypeError, ValueError):
        return 0.0
    
    # 1. 计算绝对相似度（使用新的分段映射）
    absolute_percent = _to_percent(d)
    
    # 2. 如果所有结果距离相同或差异很小，返回绝对相似度
    range_span = max_d - min_d
    if range_span < 0.01:
        return round(absolute_percent, 2)
    
    # 3. 计算相对排名相似度
    # 第一名 100%，最后一名根据绝对距离
    relative_position = (d - min_d) / range_span  # 0=最相似, 1=最不相似
    
    # 相对排名的基准：第一名从绝对相似度开始，逐步下降
    first_percent = _to_percent(min_d)
    last_percent = _to_percent(max_d)
    
    # 线性插值
    relative_percent = first_percent - relative_position * (first_percent - last_percent)
    
    # 4. 混合策略：主要使用相对排名（让结果差异更明显）
    # 但确保绝对距离很好时相似度也高
    if d < 0.8:
        # 同模态或非常好的匹配，更信任绝对距离
        weight_absolute = 0.7
    elif d < 1.2:
        # 好的跨模态匹配
        weight_absolute = 0.5
    else:
        # 一般的跨模态匹配，更看相对排名
        weight_absolute = 0.3
    
    final_percent = weight_absolute * absolute_percent + (1 - weight_absolute) * relative_percent
    
    return round(max(0.0, min(100.0, final_percent)), 2)


# 用于缓存 minvis_tracks_summary.json，避免每次检索都重复读文件
# Key: video_id -> dict
_TIME_SUMMARY_CACHE: Dict[str, Optional[dict]] = {}


def _get_time_summary(video_id: str) -> Optional[dict]:
    """
    读取并缓存视频对应的 minvis_tracks_summary.json。
    
    数据格式：
    - JSON 文件位于: data/{video_id}/photos/fullbody/{video_id}_minvis_tracks_summary.json
    - 包含 tracks 数组，每个 track 有 track_id, start_time_seconds, end_time_seconds 等字段
    """
    if not video_id:
        return None

    if video_id in _TIME_SUMMARY_CACHE:
        return _TIME_SUMMARY_CACHE[video_id]

    import glob
    
    # 主要搜索路径: data/{video_id}/photos/fullbody/
    search_dirs = [
        os.path.join(DATA_DIR, video_id, "photos", "fullbody"),
        os.path.join(DATA_DIR, video_id),  # Fallback
    ]
    
    # 搜索模式: *_minvis_tracks_summary.json
    search_patterns = [
        f"{video_id}_minvis_tracks_summary.json",
        "*_minvis_tracks_summary.json",
    ]
    
    json_path = None
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for pattern in search_patterns:
            matches = glob.glob(os.path.join(d, pattern))
            if matches:
                # 优先选择与 video_id 匹配的文件
                best_match = matches[0]
                for m in matches:
                    if os.path.basename(m).startswith(video_id):
                        best_match = m
                        break
                json_path = best_match
                break
        if json_path:
            break

    if not json_path:
        _TIME_SUMMARY_CACHE[video_id] = None
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _TIME_SUMMARY_CACHE[video_id] = data
            return data
    except Exception as e:
        print(f"Error reading summary json {json_path}: {e}")
        _TIME_SUMMARY_CACHE[video_id] = None
        return None


def _resolve_video_path(video_id: str) -> str:
    """
    根据 video_id 解析出视频文件路径。
    
    _frames格式: data/{video_id}/*.avi (视频在目录根下)
    """
    if not video_id:
        return ""

    folder_path = os.path.join(DATA_DIR, video_id)
    if not os.path.exists(folder_path):
        return ""

    # _frames格式: 视频在 DATA_DIR 根目录下
    # 对于 AvenueTrain_01_frames，视频是 AvenueTrain_01.avi
    base_name = video_id.replace("_frames", "")
    for ext in (".avi", ".AVI", ".mp4", ".MP4"):
        cand = os.path.join(folder_path, f"{base_name}{ext}")
        if os.path.exists(cand):
            return cand

    # 兜底：目录下任何视频文件
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                return os.path.join(folder_path, filename)
    except OSError:
        pass

    return ""


def _parse_video_from_image_path(image_path: str) -> Tuple[str, str]:
    """
    根据图片路径推断视频编号并匹配可用的视频文件路径。
    
    _frames格式: data/{video_id}/photos/fullbody/*.jpg
    """
    if not image_path:
        return "", ""

    video_id = ""

    # 从路径结构中解析 video_id
    try:
        abs_image_path = os.path.abspath(image_path)
        abs_data_dir = os.path.abspath(DATA_DIR)

        if os.path.normcase(abs_image_path).startswith(os.path.normcase(abs_data_dir)):
            rel_path = os.path.relpath(abs_image_path, abs_data_dir)
            parts = rel_path.split(os.sep)
            if len(parts) > 0:
                # 第一层目录即为 video_id
                video_id = parts[0]
    except Exception as e:
        print(f"Error parsing video_id from path {image_path}: {e}")

    # 如果路径解析失败，尝试从文件名解析
    if not video_id or not os.path.exists(os.path.join(DATA_DIR, video_id)):
        filename = os.path.basename(image_path)
        # 尝试匹配 _frames 格式的文件名，如 AvenueTrain_01_frames_track_0000_fullbody.jpg
        if "_frames_" in filename:
            parts = filename.split("_frames_")
            if parts:
                video_id = parts[0] + "_frames"

    if not video_id or not os.path.exists(os.path.join(DATA_DIR, video_id)):
        return "", ""

    video_path = _resolve_video_path(video_id)
    return video_id, video_path


def get_timestamps(video_id: str, track_id: int) -> Tuple[Optional[float], Optional[float]]:
    """
    从 minvis_tracks_summary.json 中读取指定 track_id 的时间戳范围。
    
    Args:
        video_id: 视频ID（即目录名，如 AvenueTrain_01_frames）
        track_id: 轨迹ID（如 0, 1, 2...）
    
    Returns:
        (start_time_seconds, end_time_seconds) 或 (None, None)
    """
    summary = _get_time_summary(video_id)
    if not summary:
        return None, None

    # JSON 结构: { "tracks": [ { "track_id": 0, "start_time_seconds": ..., "end_time_seconds": ... }, ... ] }
    tracks = summary.get("tracks", [])
    if not isinstance(tracks, list):
        return None, None

    try:
        for track in tracks:
            if isinstance(track, dict) and track.get("track_id") == track_id:
                start = track.get("start_time_seconds")
                end = track.get("end_time_seconds")
                return start, end
    except Exception as e:
        print(f"Error reading timestamp for {video_id} track_id={track_id}: {e}")

    return None, None


def _extract_timestamps(image_path: str, video_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    从图片路径中提取时间戳信息。
    
    支持的文件名格式:
    - {video_id}_track_{track_id}_fullbody.jpg (如 AvenueTrain_01_frames_track_0000_fullbody.jpg)
    
    Args:
        image_path: 图片完整路径
        video_id: 视频ID（目录名）
    
    Returns:
        (start_time_seconds, end_time_seconds) 或 (None, None)
    """
    if not video_id or not image_path:
        return None, None

    try:
        import re
        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)
        
        # 匹配格式: {video_id}_track_{track_id}_fullbody
        # 例如: AvenueTrain_01_frames_track_0000_fullbody -> track_id = 0
        # 或: 33195950@N00_7349648576_18848ca0f7_frames_track_0001_fullbody -> track_id = 1
        
        # 方法1: 使用正则表达式提取 track_XXXX
        match = re.search(r'_track_(\d+)_', stem)
        if match:
            track_id = int(match.group(1))
            return get_timestamps(video_id, track_id)
        
        # 方法2: 兼容旧格式 - 尝试从 stem 中提取数字ID
        # 例如: video_id_ped_0001 或 video_id_face_0001
        if stem.startswith(video_id):
            suffix = stem[len(video_id):]
            # suffix 可能是 _track_0000_fullbody 或 _ped_0001
            parts = suffix.split('_')
            for i, part in enumerate(parts):
                if part.isdigit():
                    track_id = int(part)
                    return get_timestamps(video_id, track_id)
                    
    except Exception as e:
        print(f"Error processing timestamps for {image_path}: {e}")

    return None, None


def _format_image_record(request: Request, record: dict, min_dist: float = None, max_dist: float = None) -> dict:
    """
    统一的图片结果结构，包含来源视频信息，便于前端按钮跳转。
    
    Args:
        request: FastAPI Request 对象
        record: LanceDB 返回的记录
        min_dist: 本次搜索结果的最小距离（用于相对排名计算）
        max_dist: 本次搜索结果的最大距离（用于相对排名计算）
    """
    image_path = record.get("image_path", "")
    video_id, video_path = _parse_video_from_image_path(image_path)
    video_url = build_video_url(request, video_path) if video_path else ""

    # 尝试提取该人脸/行人 在视频中的时间范围（供前端精准跳转/播放片段）
    start_time, end_time = _extract_timestamps(image_path, video_id)

    if start_time is not None:
        start_time = float(start_time)
    if end_time is not None:
        end_time = float(end_time)

    # Add time fragment to video URL for playback of specific segment
    # Format: #t=start,end
    if video_url and start_time is not None:
        # Use a slightly wider window (e.g. -1s, +1s) for better context, ensuring non-negative
        safe_start = max(0, start_time - 1.0)
        safe_end = end_time + 1.0 if end_time is not None else ""
        
        if safe_end:
            video_url = f"{video_url}#t={safe_start},{safe_end}"
        else:
            video_url = f"{video_url}#t={safe_start}"

    # 计算相似度：如果提供了 min_dist 和 max_dist，使用相对排名；否则使用旧方法
    distance = record.get("_distance", MAX_DISTANCE)
    if min_dist is not None and max_dist is not None:
        similarity = _to_percent_relative(distance, min_dist, max_dist)
    else:
        similarity = _to_percent(distance)

    return {
        "id": record.get("id"),
        "type": "image",
        "image_path": image_path,
        "image_url": build_image_url(request, image_path),
        "similarity": similarity,
        "video_id": video_id,
        "video_path": video_url,
        "video_url": video_url,
        "start_time": start_time,
        "end_time": end_time,
    }


def _format_text_record(request: Request, record: dict) -> dict:
    image_path = record.get("image_path", "")
    video_id, video_path = _parse_video_from_image_path(image_path)
    video_url = build_video_url(request, video_path) if video_path else ""

    start_time, end_time = _extract_timestamps(image_path, video_id)

    if start_time is not None:
        start_time = float(start_time)
    if end_time is not None:
        end_time = float(end_time)

    # Add time fragment to video URL for playback of specific segment
    # Format: #t=start,end
    if video_url and start_time is not None:
        # Use a slightly wider window (e.g. -1s, +1s) for better context, ensuring non-negative
        safe_start = max(0, start_time - 1.0)
        safe_end = end_time + 1.0 if end_time is not None else ""
        
        if safe_end:
            video_url = f"{video_url}#t={safe_start},{safe_end}"
        else:
            video_url = f"{video_url}#t={safe_start}"

    return {
        "id": record.get("id"),
        "type": "text",
        "reference_image_path": image_path,
        "reference_image_url": build_image_url(request, image_path),
        "similarity": _to_percent(record.get("_distance", MAX_DISTANCE)),
        "video_id": video_id,
        "video_path": video_url,
        "video_url": video_url,
        "start_time": start_time,
        "end_time": end_time,
    }


@app.post("/search_by_text")
async def search_by_text(
    request: Request,
    query: str = Form(...),
    limit: int = Form(5)
):
    if query is None or not str(query).strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    try:
        limit = int(limit)
    except Exception:
        raise HTTPException(status_code=400, detail="limit 参数必须为整数")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须为正数")
    if limit > 50:
        limit = 50

    vec = vector_utils.get_text_vector(query)
    if vec is None:
        raise HTTPException(status_code=500, detail="无法为查询文本生成向量")

    results = lancedb_utils.search_in_table(
        table,
        vec,
        search_column="vector_image",
        limit=limit,
    )

    # 计算距离范围（用于相对排名）
    if results:
        distances = [r.get("_distance", MAX_DISTANCE) for r in results]
        min_dist = min(distances)
        max_dist = max(distances)
    else:
        min_dist = None
        max_dist = None

    formatted = [_format_image_record(request, r, min_dist, max_dist) for r in results]

    return {"success": True, "query": query, "results": formatted}


@app.post("/search_by_image")
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5)
):
    if file is None:
        raise HTTPException(status_code=400, detail="请上传图片文件")
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="图片内容为空")
        img = Image.open(io.BytesIO(raw))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取图片，请检查文件格式")

    try:
        limit = int(limit)
    except Exception:
        raise HTTPException(status_code=400, detail="limit 参数必须为整数")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须为正数")
    if limit > 50:
        limit = 50

    vec = vector_utils.get_image_vector(img)
    if vec is None:
        raise HTTPException(status_code=500, detail="无法为上传图片生成向量")

    # 以图搜图：直接在 vector_image 列上检索人脸图片
    results = lancedb_utils.search_in_table(
        table,
        vec,
        search_column="vector_image",
        limit=limit,
    )

    # 计算距离范围（用于相对排名）
    if results:
        distances = [r.get("_distance", MAX_DISTANCE) for r in results]
        min_dist = min(distances)
        max_dist = max(distances)
    else:
        min_dist = None
        max_dist = None

    formatted = [_format_image_record(request, r, min_dist, max_dist) for r in results]

    return {"success": True, "results": formatted}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": vector_utils._model is not None}


@app.get("/system/info")
async def system_info():
    count = lancedb_utils.get_table_stats(table)
    return {
        "vector_dim": vector_dim,
        "records": count,
        "model": vector_utils.CLIP_MODEL_NAME,
        "table": table.name,
    }


# ==================== 视频流端点 ====================

def _parse_range_header(range_header: str, file_size: int) -> Tuple[int, int]:
    """
    解析Range请求头
    
    Args:
        range_header: HTTP Range头，如 "bytes=0-1023"
        file_size: 文件总大小
        
    Returns:
        (start, end) 字节范围
    """
    range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
    if not range_match:
        return 0, file_size - 1
    
    start_str, end_str = range_match.groups()
    
    if start_str and end_str:
        start = int(start_str)
        end = min(int(end_str), file_size - 1)
    elif start_str:
        start = int(start_str)
        end = file_size - 1
    elif end_str:
        # bytes=-500 表示最后500字节
        start = max(file_size - int(end_str), 0)
        end = file_size - 1
    else:
        start = 0
        end = file_size - 1
    
    return start, end


def _video_stream_generator(video_path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    """
    视频流生成器
    
    Args:
        video_path: 视频文件路径
        start: 开始字节位置
        end: 结束字节位置
        chunk_size: 块大小，默认1MB
    """
    with open(video_path, 'rb') as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk_bytes = min(chunk_size, remaining)
            data = f.read(chunk_bytes)
            if not data:
                break
            remaining -= len(data)
            yield data


@app.get("/video/{video_path:path}")
async def stream_video(
    video_path: str,
    range: Optional[str] = Header(None, alias="Range")
):
    """
    视频流端点，支持HTTP Range请求
    
    支持浏览器的视频播放，包括：
    - 拖动进度条（seek）
    - 大文件分块传输
    - 正确的Content-Type
    
    Args:
        video_path: 相对于data目录的视频路径
        range: HTTP Range请求头
        
    Returns:
        视频流响应
    """
    # 构建完整的文件路径
    full_path = os.path.join(DATA_DIR, video_path)
    full_path = os.path.normpath(full_path)
    
    # 安全检查：确保路径在DATA_DIR内
    abs_data_dir = os.path.abspath(DATA_DIR)
    abs_full_path = os.path.abspath(full_path)
    if not os.path.normcase(abs_full_path).startswith(os.path.normcase(abs_data_dir)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_path}")
    
    # 获取文件信息
    file_size = os.path.getsize(full_path)
    
    # 检测MIME类型
    content_type, _ = mimetypes.guess_type(full_path)
    if content_type is None:
        content_type = "video/mp4"  # 默认使用mp4
    
    # 处理Range请求
    if range:
        start, end = _parse_range_header(range, file_size)
        content_length = end - start + 1
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
        }
        
        return StreamingResponse(
            _video_stream_generator(full_path, start, end),
            status_code=206,
            headers=headers,
            media_type=content_type
        )
    else:
        # 没有Range请求，返回完整文件
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        
        return StreamingResponse(
            _video_stream_generator(full_path, 0, file_size - 1),
            status_code=200,
            headers=headers,
            media_type=content_type
        )


@app.get("/video-info/{video_path:path}")
async def get_video_info(video_path: str):
    """
    获取视频信息，包括编码兼容性检查
    
    Args:
        video_path: 相对于data目录的视频路径
        
    Returns:
        视频信息（编码、分辨率、帧率、时长、浏览器兼容性）
    """
    try:
        from utils.video_utils import get_video_info, is_browser_compatible
    except ImportError:
        raise HTTPException(status_code=500, detail="视频工具模块未安装")
    
    full_path = os.path.join(DATA_DIR, video_path)
    full_path = os.path.normpath(full_path)
    
    # 安全检查
    abs_data_dir = os.path.abspath(DATA_DIR)
    abs_full_path = os.path.abspath(full_path)
    if not os.path.normcase(abs_full_path).startswith(os.path.normcase(abs_data_dir)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_path}")
    
    info = get_video_info(full_path)
    if not info:
        raise HTTPException(status_code=500, detail="无法读取视频信息")
    
    info['browser_compatible'] = is_browser_compatible(full_path)
    info['path'] = video_path
    
    return {"success": True, "data": info}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)