import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.tools.agent_tools import (
    fetch_external_data,
    get_external_record,
    get_external_records,
    get_weather,
    rag_summarize,
)
from model.factory import chat_model
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


class ReactAgent:
    def __init__(self, user_id: Optional[str] = None, city: Optional[str] = None):
        self.system_prompt = load_system_prompts()
        self.max_history_turns = int(agent_conf.get("max_history_turns", 4))
        self.max_answer_chars = int(agent_conf.get("max_answer_chars", 260))
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Dict[str, str] = {
            "user_id": user_id or str(agent_conf.get("default_user_id", "1001")),
            "city": city or str(agent_conf.get("default_city", "西安")),
            "report_month": "",
            "last_intent": "",
        }
        self.metrics: Dict[str, int] = {"model_calls": 0, "tool_calls": 0, "estimated_input_chars": 0}

    def update_context(self, user_id: Optional[str] = None, city: Optional[str] = None, month: Optional[str] = None) -> None:
        if user_id:
            self.context["user_id"] = user_id
        if city:
            self.context["city"] = city
        if month:
            self.context["report_month"] = month

    def clear_memory(self) -> None:
        self.conversation_history = []
        self.context["last_intent"] = ""

    @staticmethod
    def _invoke_tool(tool: Any, payload: Optional[dict] = None) -> Any:
        payload = payload or {}
        if hasattr(tool, "invoke"):
            return tool.invoke(payload)
        if callable(tool):
            return tool(**payload) if payload else tool()
        raise TypeError(f"Unsupported tool type: {type(tool)!r}")

    @staticmethod
    def _to_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    @staticmethod
    def _extract_city(query: str) -> Optional[str]:
        query = (query or "").strip()
        patterns = [
            r"(?:今天|明天|后天|现在|实时)?([\u4e00-\u9fff]{2,8}?)(?:市)?(?:的)?(?:实时|当前|今天)?天气",
            r"(?:查一下|查询|看看)?([\u4e00-\u9fff]{2,8}?)(?:市)?(?:现在|当前)?(?:多少度|气温|温度)",
        ]
        prefixes = ("今天", "明天", "后天", "现在", "实时", "查询", "查一下", "看看")
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                city = match.group(1).strip().replace("的", "")
                for prefix in prefixes:
                    if city.startswith(prefix):
                        city = city[len(prefix):]
                if city and city not in {"天气", "气温", "温度", "多少度"}:
                    return city
        return None

    @staticmethod
    def _parse_report_query(query: str) -> Tuple[Optional[str], Optional[str], int]:
        user_match = re.search(r"(?:用户)?(\d{4})", query)
        month_match = re.search(r"(20\d{2})[年/-]?(\d{1,2})月?", query)
        trend_match = re.search(r"(?:最近|近)([一二三四五六1-6])个?月", query)
        user_id = user_match.group(1) if user_match else None
        month = None
        if month_match:
            month = f"{month_match.group(1)}-{int(month_match.group(2)):02d}"
        month_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        trend_months = 1
        if trend_match:
            raw = trend_match.group(1)
            trend_months = month_map.get(raw, int(raw) if raw.isdigit() else 3)
        elif any(word in query for word in ("趋势", "对比", "变化")):
            trend_months = 3
        return user_id, month, trend_months

    @staticmethod
    def _intent(query: str) -> str:
        weather_words = ("天气", "气温", "温度", "多少度", "下雨", "降雨", "晴天", "阴天", "潮湿")
        if any(word in query for word in weather_words):
            if any(word in query for word in ("拖地", "清扫", "扫地", "湿拖")):
                return "weather_advice"
            return "weather"
        if any(word in query for word in ("报告", "月报", "统计", "趋势", "使用情况")):
            if any(word in query for word in ("保养", "耗材", "维护", "建议")):
                return "report_advice"
            return "report"
        robot_words = (
            "扫地机器人", "扫拖", "故障", "报错", "维护", "保养", "选购", "清洁", "吸力",
            "拖地", "避障", "基站", "尘盒", "电池", "滚刷", "边刷", "水箱", "耗材", "导航",
        )
        if any(word in query for word in robot_words):
            return "rag"
        return "general"

    def _shorten(self, text: str, limit: Optional[int] = None) -> str:
        limit = limit or self.max_answer_chars
        text = (text or "").strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "..."

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            self.metrics["model_calls"] += 1
            self.metrics["estimated_input_chars"] += sum(len(item.get("content", "")) for item in messages)
            result = chat_model.invoke(messages)
            content = getattr(result, "content", None)
            return self._shorten(str(content)) if content else ""
        except Exception as exc:
            logger.error(f"[ReactAgent]模型调用失败：{exc}", exc_info=True)
            return ""

    def _tool(self, tool: Any, payload: Optional[dict] = None) -> str:
        self.metrics["tool_calls"] += 1
        return self._to_text(self._invoke_tool(tool, payload))

    def _answer_weather(self, query: str) -> str:
        city = self._extract_city(query) or self.context.get("city")
        if not city:
            return "请先提供城市，例如：西安今天天气。"
        self.context["city"] = city
        return self._shorten(self._tool(get_weather, {"city": city}), 320)

    def _format_report(self, user_id: str, month: str, record: Dict[str, str]) -> str:
        return (
            f"### {user_id} · {month} 使用月报\n"
            f"- **使用场景**：{record.get('特征', '无')}\n"
            f"- **清洁表现**：{record.get('效率', '无')}\n"
            f"- **耗材状态**：{record.get('耗材', '无')}\n"
            f"- **对比与建议**：{record.get('对比', '无')}"
        )

    def _answer_report(self, query: str) -> str:
        user_id, month, trend_months = self._parse_report_query(query)
        user_id = user_id or self.context.get("user_id")
        month = month or self.context.get("report_month")
        self.context["user_id"] = user_id

        if trend_months > 1:
            records = get_external_records(user_id, trend_months)
            if not records:
                return f"用户{user_id}暂无可用月报数据。"
            data_lines = []
            for item in reversed(records):
                record = item["record"]
                data_lines.append(
                    f"月份={item['month']}；场景={record.get('特征', '无')}；"
                    f"效率={record.get('效率', '无')}；耗材={record.get('耗材', '无')}；"
                    f"对比={record.get('对比', '无')}"
                )
            prompt = (
                f"用户ID：{user_id}\n分析范围：最近{len(records)}个月\n"
                f"用户请求：{query}\n使用数据：\n" + "\n".join(data_lines)
            )
            generated = self._call_llm([
                {"role": "system", "content": load_report_prompts()},
                {"role": "user", "content": prompt},
            ])
            if generated:
                return generated
            lines = [f"### {user_id} 最近{len(records)}个月使用趋势"]
            for item in reversed(records):
                record = item["record"]
                lines.append(f"- **{item['month']}**：{record.get('效率', '无')}；{record.get('耗材', '无')}")
            return "\n".join(lines)

        if not month:
            return f"请选择月份后再生成用户{user_id}的月报。"
        self.context["report_month"] = month
        record = get_external_record(user_id, month)
        if not record:
            return f"用户{user_id}在{month}没有使用数据，请更换用户或月份。"

        prompt = (
            f"用户ID：{user_id}\n月份：{month}\n用户请求：{query}\n"
            f"使用场景：{record.get('特征', '无')}\n"
            f"清洁表现：{record.get('效率', '无')}\n"
            f"耗材状态：{record.get('耗材', '无')}\n"
            f"同类对比：{record.get('对比', '无')}"
        )
        generated = self._call_llm([
            {"role": "system", "content": load_report_prompts()},
            {"role": "user", "content": prompt},
        ])
        return generated or self._format_report(user_id, month, record)

    def _answer_rag(self, query: str) -> str:
        return self._shorten(self._tool(rag_summarize, {"query": query}), 1600)

    def _answer_weather_advice(self, query: str) -> str:
        weather = self._answer_weather(query)
        knowledge = self._answer_rag("潮湿或降雨天气使用扫地机器人拖地要注意什么")
        prompt = (
            f"用户问题：{query}\n实时天气：{weather}\n知识库建议：{knowledge}\n"
            "请结合真实天气和知识库资料，给出具体、安全、可执行的清洁方案。"
        )
        generated = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ])
        return generated or f"{weather}\n\n### 清洁建议\n{knowledge}"

    def _answer_report_advice(self, query: str) -> str:
        report = self._answer_report(query)
        if "没有使用数据" in report or "请选择月份" in report:
            return report
        knowledge = self._answer_rag("扫地机器人耗材维护和保养建议")
        return self._shorten(f"{report}\n\n### 知识库保养参考\n{knowledge}", 2000)

    def _answer_general(self, query: str) -> str:
        history = self.conversation_history[-self.max_history_turns * 2:]
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query[:180]})
        return self._call_llm(messages) or "暂时无法回答，请换一种问法。"

    def answer(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "请输入问题。"
        intent = self._intent(query)
        self.context["last_intent"] = intent
        handlers = {
            "weather": self._answer_weather,
            "weather_advice": self._answer_weather_advice,
            "report": self._answer_report,
            "report_advice": self._answer_report_advice,
            "rag": self._answer_rag,
            "general": self._answer_general,
        }
        answer = handlers[intent](query)
        self.conversation_history.extend([
            {"role": "user", "content": query[:180]},
            {"role": "assistant", "content": self._shorten(answer, 360)},
        ])
        return answer

    def execute_stream(self, query: str):
        answer = self.answer(query)
        if answer:
            yield answer + "\n"
