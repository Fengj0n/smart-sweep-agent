import textwrap

import streamlit as st

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import list_available_months, list_available_users

st.set_page_config(
    page_title="智扫通机器人客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_STYLES = textwrap.dedent(
    """
    <style>
        :root {
            --bg-1: #eff6ff;
            --bg-2: #f8fafc;
            --surface: rgba(255, 255, 255, 0.84);
            --surface-strong: rgba(255, 255, 255, 0.96);
            --border: rgba(15, 23, 42, 0.08);
            --text: #0f172a;
            --muted: #64748b;
            --brand: #0f766e;
            --brand-2: #14b8a6;
            --brand-3: #0ea5e9;
            --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 25%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 24%),
                linear-gradient(180deg, var(--bg-2) 0%, #eef2f7 100%);
            color: var(--text);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.15rem;
            padding-bottom: 1.15rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(248, 250, 252, 0.94) 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        .shell {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            padding: 16px 18px;
            border-radius: 24px;
            border: 1px solid var(--border);
            background: var(--surface);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow);
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }

        .brand-logo {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            color: white;
            font-size: 1.3rem;
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 48%, #0ea5e9 100%);
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22);
            flex: 0 0 auto;
        }

        .brand-text {
            min-width: 0;
        }

        .brand-title {
            margin: 0;
            font-size: 1.28rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: 0.2px;
            line-height: 1.2;
        }

        .brand-subtitle {
            margin-top: 5px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.4;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            background: rgba(15, 118, 110, 0.08);
            color: #0f766e;
            font-size: 0.92rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #10b981;
            box-shadow: 0 0 0 5px rgba(16, 185, 129, 0.15);
        }

        .workspace {
            padding: 18px;
        }

        .workspace-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .workspace-title {
            margin: 0;
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--text);
        }

        .workspace-hint {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .message-wrap {
            padding-top: 4px;
        }

        div[data-testid="stChatMessage"] {
            padding: 0.12rem 0;
        }

        div[data-testid="stChatMessage"] > div {
            border-radius: 22px;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.72;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stAvatarUser"]) > div {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
            color: white;
            border: none;
            box-shadow: 0 12px 28px rgba(15, 118, 110, 0.18);
        }

        div[data-testid="stChatMessage"]:has([data-testid="stAvatarUser"]) [data-testid="stMarkdownContainer"] p {
            color: white;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stAvatarAssistant"]) > div {
            background: var(--surface-strong);
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .empty-state {
            padding: 16px 2px 2px;
            color: var(--muted);
            font-size: 0.95rem;
            text-align: center;
        }

        .sidebar-card {
            padding: 16px;
            margin-bottom: 14px;
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .sidebar-title {
            margin: 0 0 12px 0;
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
        }

        .sidebar-item {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.06);
            color: var(--muted);
            font-size: 0.92rem;
        }

        .sidebar-item:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .sidebar-item strong {
            color: var(--text);
            font-weight: 800;
        }

        div[data-testid="stChatInput"] {
            padding-top: 0.4rem;
            background: transparent;
        }

        div[data-testid="stChatInput"] textarea {
            border-radius: 18px !important;
        }

        @media (max-width: 960px) {
            .hero-stats {
                grid-template-columns: 1fr;
                min-width: 100%;
            }

            .topbar {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
    """
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "preset_prompt" not in st.session_state:
    st.session_state["preset_prompt"] = ""

users = list_available_users()
selected_user = st.session_state["agent"].context.get("user_id", users[0] if users else "1001")
if selected_user not in users and users:
    selected_user = users[0]

with st.sidebar:
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">当前上下文</div></div>', unsafe_allow_html=True)
    user_id = st.selectbox("用户", users, index=users.index(selected_user) if selected_user in users else 0)
    months = list_available_months(user_id)
    selected_month = st.session_state["agent"].context.get("report_month", "")
    month_index = months.index(selected_month) if selected_month in months else 0
    report_month = st.selectbox("报告月份", months, index=month_index if months else None)
    city = st.text_input("天气城市", value=st.session_state["agent"].context.get("city", "西安"))
    st.session_state["agent"].update_context(user_id=user_id, city=city, month=report_month)

    metrics = st.session_state["agent"].metrics
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-title">运行统计</div>
            <div class="sidebar-item"><span>消息</span><strong>{len(st.session_state['messages'])}</strong></div>
            <div class="sidebar-item"><span>模型调用</span><strong>{metrics['model_calls']}</strong></div>
            <div class="sidebar-item"><span>工具调用</span><strong>{metrics['tool_calls']}</strong></div>
            <div class="sidebar-item"><span>输入字符估算</span><strong>{metrics['estimated_input_chars']}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = [
        f"生成用户{user_id}在{report_month}的使用报告",
        f"查看用户{user_id}最近三个月趋势",
        f"{city}实时天气",
        "机器人报错怎么办？",
    ]
    for index, item in enumerate(quick_prompts):
        if st.button(item, key=f"sidebar_quick_{index}", use_container_width=True):
            st.session_state["preset_prompt"] = item
            st.rerun()

    export_text = "\n\n".join(
        f"{'用户' if message['role'] == 'user' else '客服'}：{message['content']}"
        for message in st.session_state["messages"]
    )
    st.download_button(
        "导出对话",
        data=export_text.encode("utf-8"),
        file_name="smart_sweep_chat.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not bool(st.session_state["messages"]),
    )

    if st.button("清空对话", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["preset_prompt"] = ""
        st.session_state["agent"].clear_memory()
        st.rerun()

st.markdown(
    """
    <div class="shell">
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-logo">🤖</div>
                <div class="brand-text">
                    <div class="brand-title">智扫通机器人客服</div>
                    <div class="brand-subtitle">围绕扫地机器人使用、维护、故障处理与选购建议的轻量问答界面</div>
                </div>
            </div>
            <div class="status-pill"><span class="status-dot"></span>在线服务</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="panel workspace">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="workspace-head">
        <div>
            <div class="workspace-title">对话区</div>
            <div class="workspace-hint">输入问题，系统会自动处理并返回结果</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="empty-state">
            选择一个快捷问题，或直接在下方输入你的问题。
        </div>
        """,
        unsafe_allow_html=True,
    )

quick_prompts_main = [
    "扫地机器人选购怎么选？",
    f"生成用户{user_id}在{report_month}的使用报告",
    f"{city}实时天气",
]

quick_cols = st.columns(len(quick_prompts_main))
for index, (col, item) in enumerate(zip(quick_cols, quick_prompts_main)):
    with col:
        if st.button(item, key=f"main_quick_{index}", use_container_width=True):
            st.session_state["preset_prompt"] = item
            st.rerun()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("</div>", unsafe_allow_html=True)

prompt = st.chat_input("输入你的问题")
if st.session_state["preset_prompt"]:
    prompt = st.session_state["preset_prompt"]
    st.session_state["preset_prompt"] = ""

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在整理回答..."):
            response_stream = st.session_state["agent"].execute_stream(prompt)
            response_text = st.write_stream(response_stream)

    if response_text:
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.rerun()
