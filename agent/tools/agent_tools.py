import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()
external_data: Dict[str, Dict[str, Dict[str, str]]] = {}
_weather_cache: Dict[str, Dict[str, Any]] = {}


def _fetch_json(url: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "SmartSweep/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wmo_description(code: int) -> str:
    weather_map = {
        0: "晴", 1: "大多晴朗", 2: "局部多云", 3: "多云", 45: "有雾", 48: "有雾凇",
        51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨", 56: "轻度冻毛毛雨", 57: "重度冻毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨", 66: "轻度冻雨", 67: "重度冻雨",
        71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪", 80: "阵雨", 81: "较强阵雨",
        82: "强阵雨", 85: "阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
    }
    return weather_map.get(code, "天气未知")


def _normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "user_id": (row.get("user_id") or row.get("用户ID") or "").strip().replace('"', ""),
        "month": (row.get("time") or row.get("month") or row.get("时间") or "").strip().replace('"', ""),
        "feature": (row.get("feature") or row.get("特征") or "").strip().replace('"', ""),
        "efficiency": (row.get("efficiency") or row.get("清洁效率") or "").strip().replace('"', ""),
        "consumables": (row.get("consumables") or row.get("耗材") or "").strip().replace('"', ""),
        "comparison": (row.get("comparison") or row.get("对比") or "").strip().replace('"', ""),
    }


def generate_external_data(force: bool = False) -> None:
    if external_data and not force:
        return
    external_data.clear()
    external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, "r", encoding="utf-8") as f:
        for raw_row in csv.DictReader(f):
            row = _normalize_row(raw_row)
            if not row["user_id"] or not row["month"]:
                continue
            external_data.setdefault(row["user_id"], {})[row["month"]] = {
                "特征": row["feature"],
                "效率": row["efficiency"],
                "耗材": row["consumables"],
                "对比": row["comparison"],
            }
    logger.info(f"[external_data]已加载{sum(len(item) for item in external_data.values())}条使用记录")


def list_available_users() -> List[str]:
    generate_external_data()
    return sorted(external_data.keys())


def list_available_months(user_id: Optional[str] = None) -> List[str]:
    generate_external_data()
    if user_id and user_id in external_data:
        return sorted(external_data[user_id].keys(), reverse=True)
    months = set()
    for records in external_data.values():
        months.update(records.keys())
    return sorted(months, reverse=True)


def get_external_record(user_id: str, month: str) -> Optional[Dict[str, str]]:
    generate_external_data()
    return external_data.get(user_id, {}).get(month)


def get_external_records(user_id: str, months: int = 3) -> List[Dict[str, Any]]:
    generate_external_data()
    result: List[Dict[str, Any]] = []
    for month in sorted(external_data.get(user_id, {}).keys(), reverse=True)[:months]:
        result.append({"month": month, "record": external_data[user_id][month]})
    return result


@tool
def rag_summarize(query: str) -> str:
    """从向量存储中检索扫地机器人资料并返回简短回答。"""
    return rag.rag_summarize(query)


@tool
def get_weather(city: str) -> str:
    """查询指定城市当前实时天气，不使用模拟数据。"""
    city = (city or "").strip()
    if not city:
        return "请提供要查询的城市。"

    cache_seconds = int(agent_conf.get("weather_cache_seconds", 600))
    cached = _weather_cache.get(city)
    if cached and time.time() - float(cached["timestamp"]) < cache_seconds:
        return str(cached["value"])

    try:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(city)}&count=5&language=zh&format=json&countryCode=CN"
        )
        results = (_fetch_json(geo_url) or {}).get("results") or []
        if not results:
            return f"未找到城市“{city}”，请检查城市名称。"

        location = results[0]
        place_name = location.get("name") or city
        admin1 = location.get("admin1") or ""
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={location['latitude']}&longitude={location['longitude']}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            "weather_code,wind_speed_10m,wind_direction_10m,precipitation"
            "&timezone=Asia%2FShanghai"
        )
        current = (_fetch_json(weather_url) or {}).get("current") or {}
        required = ("temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m", "time")
        if any(current.get(key) is None for key in required):
            return f"{place_name}实时天气数据暂时不完整。"

        desc = _wmo_description(int(current["weather_code"]))
        place = f"{admin1}{place_name}" if admin1 and admin1 not in place_name else place_name
        result = (
            f"{place}实时天气：{desc}，{current['temperature_2m']}℃，"
            f"体感{current.get('apparent_temperature', '-')}℃，湿度{current['relative_humidity_2m']}%，"
            f"风速{current['wind_speed_10m']}km/h，降水{current.get('precipitation', 0)}mm。"
            f"更新时间：{str(current['time']).replace('T', ' ')}。"
        )
        _weather_cache[city] = {"timestamp": time.time(), "value": result}
        return result
    except Exception as exc:
        logger.error(f"[get_weather]实时天气查询失败：{city}，{exc}", exc_info=True)
        return f"{city}实时天气暂时无法获取，请稍后重试。"


@tool
def get_user_location() -> str:
    """返回项目配置的默认城市，不进行随机选择。"""
    return str(agent_conf.get("default_city", "西安"))


@tool
def get_user_id() -> str:
    """返回项目配置的默认用户ID，不进行随机选择。"""
    return str(agent_conf.get("default_user_id", "1001"))


@tool
def get_current_month() -> str:
    """返回系统当前月份，格式为YYYY-MM。"""
    return datetime.now().strftime("%Y-%m")


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """查询指定用户指定月份的扫地机器人使用记录。"""
    record = get_external_record(user_id, month)
    if not record:
        logger.warning(f"[fetch_external_data]用户{user_id}在{month}无使用记录")
        return ""
    return (
        f"用户{user_id}，月份{month}；特征：{record['特征']}；"
        f"效率：{record['效率']}；耗材：{record['耗材']}；对比：{record['对比']}"
    )


@tool
def fill_context_for_report() -> str:
    """标记当前任务为报告生成。"""
    return "report"
