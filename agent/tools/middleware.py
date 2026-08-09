from typing import Any, Callable
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts
try:
    from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt
except Exception:
    def wrap_tool_call(func):
        return func

    def before_model(func):
        return func

    def dynamic_prompt(func):
        return func


@wrap_tool_call
def monitor_tool(request: Any, handler: Callable[[Any], Any]) -> Any:
    """监控工具调用、记录日志，并在报告工具触发时切换上下文标记。"""
    tool_call = getattr(request, "tool_call", {}) or {}
    tool_name = tool_call.get("name", "unknown")
    tool_args = tool_call.get("args", {})

    logger.info(f"[tool monitor]执行工具：{tool_name}")
    logger.info(f"[tool monitor]传入参数：{tool_args}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{tool_name}调用成功")

        if tool_name == "fill_context_for_report":
            runtime = getattr(request, "runtime", None)
            if runtime is not None and hasattr(runtime, "context"):
                runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{tool_name}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(state: Any, runtime: Any):
    """模型调用前记录日志。"""
    messages = []
    if isinstance(state, dict):
        messages = state.get("messages", []) or []

    if messages:
        logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息。")
        last_message = messages[-1]
        content = getattr(last_message, "content", None)
        if content is None and isinstance(last_message, dict):
            content = last_message.get("content", "")
        logger.debug(f"[log_before_model]{type(last_message).__name__} | {str(content).strip()}")

    return None


@dynamic_prompt
def report_prompt_switch(request: Any):
    """根据 runtime.context 中的 report 标记动态切换提示词。"""
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", {}) if runtime is not None else {}
    if context.get("report", False):
        return load_report_prompts()
    return load_system_prompts()
