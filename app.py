import streamlit as st
import os
import sys
import json
import pandas as pd
import random
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine
import config

st.set_page_config(
    page_title="政务问答系统",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="正在加载模型，请稍候...")
def get_rag_engine():
    """全局缓存的 RAG 引擎实例，只加载一次"""
    engine = RAGEngine()
    engine.initialize()
    return engine

rag_engine = get_rag_engine()

if 'feedback_list' not in st.session_state:
    st.session_state.feedback_list = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = "智能问答"

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 12847

if 'hot_questions' not in st.session_state:
    st.session_state.hot_questions = [
        {"question": "如何办理居住证续签？", "category": "户籍办理"},
        {"question": "医保异地就医如何报销？", "category": "医保社保"},
        {"question": "公积金提取需要哪些材料？", "category": "公积金"},
        {"question": "新生儿落户登记流程？", "category": "户籍办理"},
        {"question": "如何办理个体工商户注册？", "category": "企业开办"},
        {"question": "二手房交易过户流程？", "category": "不动产"},
    ]

if 'recent_updates' not in st.session_state:
    st.session_state.recent_updates = [
        {"title": "关于调整住房公积金缴存比例的通知", "date": "2024-12-15", "status": "已发布"},
        {"title": "2024年度社保缴费基数公布", "date": "2024-12-12", "status": "解析中"},
        {"title": "居住证办理流程优化公告", "date": "2024-12-10", "status": "待审核"},
        {"title": "新生儿落户登记指南（修订版）", "date": "2024-12-08", "status": "解析失败"},
        {"title": "企业开办一网通办操作规程", "date": "2024-12-05", "status": "已发布"},
    ]


def init_engine():
    pass


def load_qa_history():
    """从本地文件加载问答历史记录，并清理超过24小时的过期记录"""
    if not os.path.exists(config.QA_HISTORY_FILE):
        return []
    try:
        with open(config.QA_HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        cutoff = datetime.now() - timedelta(hours=config.QA_HISTORY_EXPIRE_HOURS)
        valid = [item for item in history
                 if datetime.fromisoformat(item.get('timestamp', '2000-01-01T00:00:00')) > cutoff]
        if len(valid) < len(history):
            save_qa_history(valid)
        return valid
    except Exception as e:
        print(f"加载历史记录失败: {e}")
        return []


def save_qa_history(history_list):
    """保存问答历史记录到本地文件"""
    try:
        os.makedirs(os.path.dirname(config.QA_HISTORY_FILE), exist_ok=True)
        with open(config.QA_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存历史记录失败: {e}")


def classify_question(question):
    """根据问题关键词分类，返回分类名称"""
    categories = {
        "户籍办理": ["户口", "户籍", "落户", "迁", "迁入", "迁出", "居住", "居住证", "身份证", "改名", "更名"],
        "医保社保": ["医保", "社保", "保险", "医疗", "报销", "就医", "养老", "失业", "工伤", "生育"],
        "公积金": ["公积金", "住房公积金", "缴存", "提取", "贷款", "买房", "购房"],
        "不动产": ["房产", "房屋", "不动产", "过户", "房产证", "抵押", "登记", "二手房", "买卖", "契税", "税"],
        "企业开办": ["企业", "公司", "工商", "注册", "营业执照", "开办", "经营", "注销", "变更"],
        "税务财税": ["税", "税务", "发票", "纳税", "个税", "增值税", "财务", "报税"],
        "教育": ["教育", "学校", "上学", "入学", "转学", "学位", "教师", "高考", "中考", "学生"],
        "交通出行": ["交通", "出行", "驾照", "驾驶证", "车辆", "车牌", "违章", "公交", "地铁"],
    }
    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in question:
                return cat
    return "其他"


def append_qa_record(question, answer, strategy, max_score, retrieval_info, references, response_time=0.0, policy_count=0, faq_count=0, ticket_count=0, rerank_count=0):
    """追加一条问答记录到历史"""
    history = load_qa_history()
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "strategy": strategy,
        "max_score": max_score,
        "retrieval_info": retrieval_info,
        "references": [
            {"content": r.content, "source": r.source,
             "section": r.section, "score": r.score, "kb_type": r.kb_type}
            for r in references
        ],
        "response_time": response_time,
        "category": classify_question(question),
        "policy_count": policy_count,
        "faq_count": faq_count,
        "ticket_count": ticket_count,
        "rerank_count": rerank_count
    }
    history.append(record)
    save_qa_history(history)
    return history


def render_gov_css():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
    }
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #f8f9fb 0%, #eef2f7 100%);
    }
    .stSidebar .stButton > button {
        border: none !important;
        background: transparent !important;
        color: #4a5568 !important;
        text-align: left !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        justify-content: flex-start !important;
    }
    .stSidebar .stButton > button:hover {
        background: rgba(24, 119, 242, 0.08) !important;
        color: #1877f2 !important;
    }
    .stSidebar .stButton > button[kind="primary"] {
        background: rgba(24, 119, 242, 0.12) !important;
        color: #1877f2 !important;
        border-left: 3px solid #1877f2 !important;
        border-radius: 0 8px 8px 0 !important;
    }
    .top-header {
        background: #ffffff;
        border-bottom: 1px solid #e8edf2;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-header h2 {
        margin: 0;
        color: #1a202c;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .stat-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e8edf2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-label {
        color: #718096;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        color: #1a202c;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stat-change {
        font-size: 0.875rem;
        font-weight: 500;
    }
    .stat-change.up {
        color: #38a169;
    }
    .stat-change.down {
        color: #e53e3e;
    }
    .content-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e8edf2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .card-title {
        color: #1a202c;
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        background: #e8f4fd;
        color: #1877f2;
        margin-right: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-published { background: #e6f7ed; color: #38a169; }
    .status-parsing { background: #e8f4fd; color: #1877f2; }
    .status-pending { background: #fff8e6; color: #d69e2e; }
    .status-failed { background: #fde8e8; color: #e53e3e; }
    .faq-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e8edf2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .faq-title {
        color: #1a202c;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .faq-content {
        color: #4a5568;
        font-size: 0.875rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }
    .nav-section-title {
        color: #a0aec0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0 1rem;
        margin: 1.5rem 0 0.5rem 0;
    }
    .sidebar-logo {
        width: 220px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .sidebar-group-title {
        width: 220px;
        height: 42px;
        display: flex;
        align-items: center;
        padding-left: 1rem;
        color: #a0aec0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 auto;
    }
    .admin-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 21rem;
        padding: 1rem;
        background: #f8f9fb;
        border-top: 1px solid #e8edf2;
        display: flex;
        align-items: center;
        color: #4a5568;
        font-size: 0.875rem;
    }
    .admin-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #cbd5e0;
        margin-right: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-weight: 600;
    }
    .trend-bar {
        background: #cfe3f7;
        border-radius: 6px 6px 0 0;
        min-width: 32px;
    }
    .trend-bar:hover {
        background: #1877f2;
    }
    /* 智能问答页面 */
    .qa-page-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a202c;
    }
    /* 聊天气泡定制 */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: #1877f2 !important;
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: #e8f4fd !important;
        color: #1877f2 !important;
    }
    </style>
    """, unsafe_allow_html=True)


render_gov_css()

init_engine()


with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="width: 36px; height: 36px; background: #1877f2; border-radius: 8px; 
                    display: flex; align-items: center; justify-content: center; 
                    margin-right: 0.75rem; color: white; font-weight: 700; flex-shrink: 0;">
            政
        </div>
        <div style="font-size: 1.125rem; font-weight: 700; color: #1a202c; white-space: nowrap;">
            政务问答系统
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title">主要功能</div>', unsafe_allow_html=True)
    main_pages = ["智能问答", "数据概览", "知识库管理", "常见问题"]
    for page in main_pages:
        icon_map = {
            "智能问答": "💬",
            "数据概览": "📊",
            "知识库管理": "�",
            "常见问题": "❓"
        }
        btn_type = "primary" if st.session_state.current_page == page else "secondary"
        if st.button(f"{icon_map[page]} {page}", use_container_width=True, type=btn_type):
            st.session_state.current_page = page
            st.rerun()

    st.markdown('<div class="sidebar-group-title">管理</div>', unsafe_allow_html=True)
    admin_pages = ["反馈记录", "系统管理"]
    for page in admin_pages:
        icon_map = {
            "反馈记录": "📝",
            "系统管理": "⚙️"
        }
        btn_type = "primary" if st.session_state.current_page == page else "secondary"
        if st.button(f"{icon_map[page]} {page}", use_container_width=True, type=btn_type):
            st.session_state.current_page = page
            st.rerun()

    st.markdown("""
    <div class="admin-footer">
        <div class="admin-avatar">管</div>
        <div>管理员</div>
    </div>
    """, unsafe_allow_html=True)


def render_top_header(title):
    col1, col2, col3 = st.columns([6, 3, 1])
    with col1:
        st.markdown(f"<div class='top-header' style='border-bottom:none;padding-left:0;'><h2>{title}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.text_input("搜索", placeholder="🔍 搜索问题或文件...", label_visibility="collapsed", key=f"search_{title}")
    with col3:
        st.markdown("<div style='text-align:right;font-size:1.25rem;'>🔔</div>", unsafe_allow_html=True)


if st.session_state.current_page == "数据概览":
    render_top_header("数据概览")

    history = load_qa_history()
    total_qa = len(history)

    if history:
        now = datetime.now()
        this_month = [h for h in history if datetime.fromisoformat(h['timestamp']).month == now.month and datetime.fromisoformat(h['timestamp']).year == now.year]
        last_month = [h for h in history if datetime.fromisoformat(h['timestamp']).month == (now.month - 1 if now.month > 1 else 12) and datetime.fromisoformat(h['timestamp']).year == (now.year if now.month > 1 else now.year - 1)]
        if last_month:
            qa_change = round((len(this_month) - len(last_month)) / len(last_month) * 100, 1)
            qa_change_text = f"+{qa_change}% 较上月" if qa_change >= 0 else f"{qa_change}% 较上月"
            qa_change_class = "up" if qa_change >= 0 else "down"
        else:
            qa_change_text = f"+{len(this_month)} 本月"
            qa_change_class = "up"
    else:
        qa_change_text = "暂无数据"
        qa_change_class = "up"

    policy_files = rag_engine.get_files_by_kb("policy")
    faq_files = rag_engine.get_files_by_kb("faq")
    ticket_files = rag_engine.get_files_by_kb("ticket")
    total_files = len(policy_files) + len(faq_files) + len(ticket_files)

    file_changes_this_week = 0
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    all_file_dirs = [config.POLICIES_DIR, config.FAQ_DIR, config.TICKETS_DIR]
    for d in all_file_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    mtime = datetime.fromtimestamp(os.path.getmtime(fp))
                    if mtime >= week_ago:
                        file_changes_this_week += 1

    response_times = [h.get('response_time', 0) for h in history if h.get('response_time', 0) > 0]
    if response_times:
        avg_resp = round(sum(response_times) / len(response_times), 2)
        recent_resp = response_times[-min(10, len(response_times)):]
        older_resp = response_times[-min(20, len(response_times)):-min(10, len(response_times))] if len(response_times) > 10 else []
        if older_resp:
            old_avg = sum(older_resp) / len(older_resp)
            new_avg = sum(recent_resp) / len(recent_resp)
            if old_avg > 0:
                resp_change = round((new_avg - old_avg) / old_avg * 100, 1)
                resp_change_text = f"{resp_change}% 优化" if resp_change < 0 else f"+{resp_change}%"
                resp_change_class = "down" if resp_change < 0 else "up"
            else:
                resp_change_text = f"{avg_resp}s"
                resp_change_class = "up"
        else:
            resp_change_text = "新数据"
            resp_change_class = "up"
    else:
        avg_resp = 0
        resp_change_text = "暂无数据"
        resp_change_class = "up"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">累计问答</div>
            <div class="stat-value">{total_qa:,}</div>
            <div class="stat-change {qa_change_class}">{qa_change_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">文件变动</div>
            <div class="stat-value">{total_files}</div>
            <div class="stat-change up">+{file_changes_this_week} 本周新增</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        resp_display = f"{avg_resp}s" if avg_resp > 0 else "--"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">平均响应</div>
            <div class="stat-value">{resp_display}</div>
            <div class="stat-change {resp_change_class}">{resp_change_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">热门问题</div>', unsafe_allow_html=True)

        question_count = {}
        question_category = {}
        for h in history:
            q = h['question']
            cat = h.get('category', classify_question(q))
            if q in question_count:
                question_count[q] += 1
            else:
                question_count[q] = 1
                question_category[q] = cat

        if question_count:
            sorted_questions = sorted(question_count.items(), key=lambda x: x[1], reverse=True)[:10]
            for q, count in sorted_questions:
                cat = question_category.get(q, '其他')
                times_display = f'<span style="color:#a0aec0;font-size:0.75rem;float:right;">{count}次</span>' if count > 1 else ''
                st.markdown(f"""
                <div style="padding: 0.75rem 0; border-bottom: 1px solid #f0f2f5;">
                    <div style="color: #2d3748; margin-bottom: 0.5rem;">
                        {q}{times_display}
                    </div>
                    <span class="tag">{cat}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#a0aec0;padding:2rem 0;text-align:center;font-size:0.9rem;'>暂无问答数据</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">近期更新</div>', unsafe_allow_html=True)

        all_files_info = []
        kb_dir_map = [
            (config.POLICIES_DIR, "政策知识库", "已发布"),
            (config.FAQ_DIR, "FAQ库", "已发布"),
            (config.TICKETS_DIR, "工单库", "已发布"),
        ]
        for dir_path, kb_name, status in kb_dir_map:
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    fp = os.path.join(dir_path, f)
                    if os.path.isfile(fp):
                        mtime = datetime.fromtimestamp(os.path.getmtime(fp))
                        all_files_info.append({
                            "title": f,
                            "date": mtime.strftime("%Y-%m-%d"),
                            "status": status,
                            "kb": kb_name,
                            "mtime": mtime
                        })

        all_files_info.sort(key=lambda x: x['mtime'], reverse=True)
        recent_files = all_files_info[:10]

        if recent_files:
            for item in recent_files:
                status_class = {
                    "已发布": "status-published",
                    "解析中": "status-parsing",
                    "待审核": "status-pending",
                    "解析失败": "status-failed"
                }.get(item['status'], "status-published")
                st.markdown(f"""
                <div style="padding: 0.75rem 0; border-bottom: 1px solid #f0f2f5;">
                    <div style="color: #2d3748; margin-bottom: 0.5rem;">{item['title']}</div>
                    <span class="status-badge {status_class}">{item['status']}</span>
                    <span style="color:#a0aec0;font-size:0.75rem;margin-left:0.5rem;">{item['kb']} · {item['date']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#a0aec0;padding:2rem 0;text-align:center;font-size:0.9rem;'>暂无文件数据</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">近7日问答趋势</div>', unsafe_allow_html=True)

    now = datetime.now()
    day_labels = []
    day_values = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        day_labels.append(d.strftime("%a"))
        count = len([h for h in history if datetime.fromisoformat(h['timestamp']).date() == d.date()])
        day_values.append(count)

    if day_values and max(day_values) > 0:
        max_val = max(day_values)
        cols = st.columns(7)
        for i, (label, val) in enumerate(zip(day_labels, day_values)):
            with cols[i]:
                height = max(int(val / max_val * 160), 4) if max_val > 0 else 4
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 200px;">
                    <div style="color:#4a5568;font-size:0.75rem;margin-bottom:0.25rem;">{val}</div>
                    <div class="trend-bar" style="width: 80%; height: {height}px;"></div>
                    <div style="margin-top: 0.5rem; color: #718096; font-size: 0.875rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#a0aec0;padding:2rem 0;text-align:center;font-size:0.9rem;'>暂无趋势数据</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.current_page == "知识库管理":
    render_top_header("知识库管理")

    ocr_status = "✅ OCR图片识别：已启用" if config.ENABLE_OCR else "⏸️ OCR图片识别：已关闭"
    ocr_class = "status-published" if config.ENABLE_OCR else "status-failed"
    st.markdown(f"<span class='status-badge {ocr_class}'>{ocr_status}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    kb_tabs = ["政策知识库", "工单知识库", "FAQ库"]
    kb_types = ["policy", "ticket", "faq"]

    if "current_kb_tab" not in st.session_state:
        st.session_state.current_kb_tab = "政策知识库"

    tab_cols = st.columns([1, 1, 1, 5])
    for i, tab_name in enumerate(kb_tabs):
        with tab_cols[i]:
            btn_type = "primary" if st.session_state.current_kb_tab == tab_name else "secondary"
            if st.button(tab_name, use_container_width=True, type=btn_type, key=f"kb_tab_{tab_name}"):
                st.session_state.current_kb_tab = tab_name
                st.rerun()

    current_kb_name = st.session_state.current_kb_tab
    current_kb_type = kb_types[kb_tabs.index(current_kb_name)]

    st.markdown("<br>", unsafe_allow_html=True)

    search_col, upload_col = st.columns([3, 2])
    with search_col:
        search_keyword = st.text_input(
            "",
            placeholder="🔍 请输入文件名",
            label_visibility="collapsed",
            key="kb_file_search"
        ).strip()
    with upload_col:
        up_btn_col, batch_up_col = st.columns(2)
        with up_btn_col:
            st.button("📤 上传文件", type="primary", use_container_width=True, key="show_upload_dialog")
        with batch_up_col:
            st.button("📁 批量导入", use_container_width=True, key="show_batch_upload")

    st.markdown("<br>", unsafe_allow_html=True)

    all_files = rag_engine.get_files_by_kb(current_kb_type)

    if search_keyword:
        filtered = [f for f in all_files if search_keyword.lower() in f["name"].lower()]
        if not filtered:
            st.warning(f"⚠️ 该文件不存在（未找到包含「{search_keyword}」的文件）")
        display_files = filtered
    else:
        display_files = all_files

    if not search_keyword or (search_keyword and display_files):
        st.markdown("<br>", unsafe_allow_html=True)
        action_cols_top = st.columns([1, 1, 1, 5])
        with action_cols_top[0]:
            if st.button("全选", use_container_width=True, key=f"select_all_{current_kb_type}"):
                for idx in range(len(display_files)):
                    st.session_state[f"kb_chk_{current_kb_type}_{idx}"] = True
                st.rerun()
        with action_cols_top[1]:
            if st.button("取消全选", use_container_width=True, key=f"deselect_all_{current_kb_type}"):
                for idx in range(len(display_files)):
                    st.session_state[f"kb_chk_{current_kb_type}_{idx}"] = False
                st.rerun()
        with action_cols_top[2]:
            st.markdown(f"<div style='padding: 0.5rem; color: #718096;'>共 {len(display_files)} 个文件</div>", unsafe_allow_html=True)

        st.markdown('<div class="content-card" style="padding: 0; overflow: hidden;">', unsafe_allow_html=True)

        header_cols = st.columns([0.5, 4, 2, 1.5, 1.5])
        headers = ["", "文件名称", "大小", "更新日期", "操作"]
        for col, h in zip(header_cols, headers):
            with col:
                st.markdown(f"<div style='padding: 1rem; border-bottom: 1px solid #e8edf2; font-weight: 600; color: #4a5568; background: #f8f9fb;'>{h}</div>", unsafe_allow_html=True)

        selected_files = []
        for idx, finfo in enumerate(display_files):
            size_kb = finfo["size"] / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"

            row_cols = st.columns([0.5, 4, 2, 1.5, 1.5])
            with row_cols[0]:
                default_val = st.session_state.get(f"kb_chk_{current_kb_type}_{idx}", False)
                checked = st.checkbox("选择", key=f"kb_chk_{current_kb_type}_{idx}", label_visibility="collapsed", value=default_val)
                if checked:
                    selected_files.append(finfo["name"])
            with row_cols[1]:
                st.markdown(f"<div style='padding: 1rem 0; color: #2d3748;'>📄 {finfo['name']}</div>", unsafe_allow_html=True)
            with row_cols[2]:
                st.markdown(f"<div style='padding: 1rem 0; color: #4a5568;'>{size_str}</div>", unsafe_allow_html=True)
            with row_cols[3]:
                st.markdown(f"<div style='padding: 1rem 0; color: #4a5568;'>{finfo['update_time'][:10]}</div>", unsafe_allow_html=True)
            with row_cols[4]:
                dl_col, del_col = st.columns(2)
                with dl_col:
                    with open(finfo["path"], "rb") as f:
                        st.download_button(
                            "下载",
                            data=f.read(),
                            file_name=finfo["name"],
                            use_container_width=True,
                            key=f"dl_{current_kb_type}_{idx}"
                        )
                with del_col:
                    if st.button("删除", key=f"del_{current_kb_type}_{idx}", use_container_width=True):
                        if rag_engine.delete_file_from_kb(current_kb_type, finfo["name"]):
                            st.success(f"已删除：{finfo['name']}")
                            st.rerun()
                        else:
                            st.error("删除失败")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        action_cols = st.columns([1, 1, 3])
        with action_cols[0]:
            if st.button("批量删除", type="secondary", use_container_width=True, key=f"batch_del_{current_kb_type}"):
                if selected_files:
                    for fname in selected_files:
                        rag_engine.delete_file_from_kb(current_kb_type, fname)
                    st.success(f"已删除 {len(selected_files)} 个文件")
                    st.rerun()
                else:
                    st.warning("请先选择要删除的文件")
        with action_cols[1]:
            rebuild_label = {
                "policy": "重建政策库索引",
                "ticket": "重建工单库索引",
                "faq": "重建FAQ库索引"
            }
            if st.button("🔄 重建索引", type="primary", use_container_width=True, key=f"rebuild_{current_kb_type}"):
                with st.spinner("正在构建索引..."):
                    if current_kb_type == "policy":
                        count = rag_engine.build_policy_index()
                    else:
                        kb_dir = rag_engine.get_kb_dir(current_kb_type)
                        count = rag_engine.build_csv_index(current_kb_type, kb_dir)
                st.success(f"索引构建完成，共 {count} 条记录")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(f"📤 上传文件到{current_kb_name}", expanded=False):
        if current_kb_type == "policy":
            ext_list = [e.lstrip('.') for e in config.SUPPORTED_DOC_EXTENSIONS]
            type_hint = "PDF, DOCX, DOC, TXT, MD 等"
        else:
            ext_list = ["csv"]
            type_hint = "CSV 格式（需包含 question, answer 列）"

        st.info(f"支持的文件类型：{type_hint}")
        uploaded_files = st.file_uploader(
            f"选择要上传到{current_kb_name}的文件",
            type=ext_list,
            accept_multiple_files=True,
            key="kb_upload_files"
        )
        if uploaded_files:
            if st.button(f"确认上传到{current_kb_name}", type="primary"):
                with st.spinner("上传中..."):
                    saved = rag_engine.save_uploaded_files_to_kb(uploaded_files, current_kb_type)
                st.success(f"已上传 {len(saved)} 个文件到{current_kb_name}")
                st.info("请记得重建索引")
                st.rerun()

    with st.expander(f"📁 批量导入文件夹到{current_kb_name}", expanded=False):
        st.info(f"将本地文件夹中的所有 {type_hint} 文件批量导入到{current_kb_name}")
        folder_path = st.text_input("请输入本地文件夹完整路径", key="batch_folder_path")
        if st.button(f"开始导入到{current_kb_name}", type="primary"):
            if not folder_path or not os.path.exists(folder_path):
                st.error("请输入有效的文件夹路径")
            else:
                if current_kb_type == "policy":
                    target_exts = set(config.SUPPORTED_DOC_EXTENSIONS)
                else:
                    target_exts = {".csv"}
                kb_dir = rag_engine.get_kb_dir(current_kb_type)
                os.makedirs(kb_dir, exist_ok=True)
                count = 0
                for root, dirs, files in os.walk(folder_path):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in target_exts:
                            src = os.path.join(root, f)
                            dst = os.path.join(kb_dir, f)
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
                                count += 1
                st.success(f"批量导入完成，共新增 {count} 个文件")
                st.info("请记得重建索引")
                st.rerun()


elif st.session_state.current_page == "常见问题":
    render_top_header("常见问题")

    search_keyword = st.text_input("搜索", placeholder="🔍 输入关键词搜索常见问题...", label_visibility="collapsed", key="faq_search").strip()

    categories = ["全部", "户籍办理", "医保社保", "公积金", "企业开办", "不动产"]
    if "faq_current_cat" not in st.session_state:
        st.session_state.faq_current_cat = "全部"
    if "faq_page" not in st.session_state:
        st.session_state.faq_page = 1
    FAQ_PAGE_SIZE = 10

    cat_cols = st.columns(len(categories))
    for i, cat in enumerate(categories):
        with cat_cols[i]:
            btn_type = "primary" if st.session_state.faq_current_cat == cat else "secondary"
            if st.button(cat, key=f"cat_{cat}", use_container_width=True, type=btn_type):
                st.session_state.faq_current_cat = cat
                st.session_state.faq_page = 1
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    faq_data = []
    if os.path.exists(config.FAQ_DIR):
        try:
            for fname in os.listdir(config.FAQ_DIR):
                if fname.lower().endswith('.csv'):
                    fpath = os.path.join(config.FAQ_DIR, fname)
                    df = pd.read_csv(fpath)
                    for _, row in df.iterrows():
                        q_text = str(row.get('question', ''))
                        faq_data.append({
                            "question": q_text,
                            "answer": str(row.get('answer', '')),
                            "category": classify_question(q_text)
                        })
        except Exception:
            pass

    if not faq_data:
        faq_data = [
            {"question": "如何办理居住证续签？", "answer": "居住证有效期届满前30日内，可前往居住地公安派出所或社区服务中心办理续签手续，需携带本人身份证及居住证明材料。", "category": "户籍办理"},
            {"question": "医保异地就医如何报销？", "answer": "参保人员跨省异地就医前，需在参保地医保经办机构办理异地就医备案登记，选择已开通异地结算的定点医疗机构就诊。", "category": "医保社保"},
            {"question": "公积金提取需要哪些材料？", "answer": "职工提取住房公积金需提供身份证、公积金提取申请表、相关证明材料原件及复印件，具体材料因提取原因而异。", "category": "公积金"},
            {"question": "新生儿落户登记流程？", "answer": "新生婴儿落户需在出生后一个月内，由父亲或母亲携带出生医学证明、父母身份证、结婚证到户籍所在地派出所办理。", "category": "户籍办理"},
            {"question": "企业注册需要哪些材料？", "answer": "注册公司需提供：公司名称、公司章程、股东身份证明、注册地址证明、法定代表人任职文件等，可通过线上或线下渠道提交申请。", "category": "企业开办"},
            {"question": "房产证丢失怎么补办？", "answer": "权利人需持身份证到不动产登记中心申请挂失，在指定报纸刊登遗失声明，公告期满后可申请补发不动产权证书。", "category": "不动产"},
            {"question": "社保断缴有什么影响？", "answer": "社保断缴可能影响医保报销、养老保险累计年限、购房购车资格等，建议及时办理续缴手续。", "category": "医保社保"},
            {"question": "如何重置密码？", "answer": "登录页面点击\"忘记密码\"，按照提示输入注册邮箱，系统会发送重置链接到您的邮箱，点击链接即可设置新密码。密码长度要求8-20位，需包含字母和数字。", "category": "其他"},
            {"question": "系统支持哪些浏览器？", "answer": "系统支持Chrome、Firefox、Edge、Safari等主流浏览器，建议使用最新版本以获得最佳体验。IE浏览器不再支持。", "category": "其他"},
            {"question": "如何修改个人信息？", "answer": "登录后点击右上角头像，选择\"个人中心\"，在\"基本信息\"页面可以修改姓名、手机号、邮箱等个人信息。", "category": "其他"},
            {"question": "忘记账号怎么办？", "answer": "如果忘记账号，可以通过注册时绑定的手机号或邮箱进行找回。在登录页面点击\"忘记账号\"，按照提示操作即可。", "category": "其他"},
            {"question": "系统登录失败怎么办？", "answer": "请检查以下几点：1.用户名和密码是否正确；2.网络连接是否正常；3.账号是否被锁定；4.浏览器Cookie是否开启。如仍无法解决，请联系客服。", "category": "其他"},
            {"question": "如何联系客服？", "answer": "您可以通过以下方式联系客服：1.在线客服（工作日9:00-18:00）；2.客服热线：400-xxx-xxxx；3.邮箱：support@example.com", "category": "其他"},
            {"question": "如何开通新账号？", "answer": "企业版账号按年付费，付费成功后有效期为一年。到期前30天会提醒续约。如有疑问请联系客服。", "category": "其他"},
            {"question": "系统有使用期限吗？", "answer": "企业版账号按年付费，付费成功后有效期为一年。到期前30天会提醒续约。如有疑问请联系客服。", "category": "其他"},
        ]

    if st.session_state.faq_current_cat != "全部":
        faq_data = [f for f in faq_data if f["category"] == st.session_state.faq_current_cat]

    if search_keyword:
        faq_data = [f for f in faq_data if search_keyword.lower() in f["question"].lower() or search_keyword.lower() in f["answer"].lower()]
        st.session_state.faq_page = 1

    if not faq_data:
        st.info("暂无匹配的常见问题")
    else:
        total_count = len(faq_data)
        total_pages = (total_count + FAQ_PAGE_SIZE - 1) // FAQ_PAGE_SIZE

        if st.session_state.faq_page > total_pages:
            st.session_state.faq_page = total_pages
        if st.session_state.faq_page < 1:
            st.session_state.faq_page = 1

        start_idx = (st.session_state.faq_page - 1) * FAQ_PAGE_SIZE
        end_idx = min(start_idx + FAQ_PAGE_SIZE, total_count)
        page_data = faq_data[start_idx:end_idx]

        cols = st.columns(2)
        for i, item in enumerate(page_data):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="faq-card">
                    <div class="faq-title">{item['question']}</div>
                    <div class="faq-content">{item['answer']}</div>
                    <span class="tag">{item['category']}</span>
                </div>
                """, unsafe_allow_html=True)

        if total_pages > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            pag_col1, pag_col2, pag_col3, pag_col4, pag_col5 = st.columns([1, 1, 2, 1, 1])
            with pag_col1:
                if st.button("⏮️ 首页", use_container_width=True, disabled=(st.session_state.faq_page == 1), key="faq_first"):
                    st.session_state.faq_page = 1
                    st.rerun()
            with pag_col2:
                if st.button("⬅️ 上一页", use_container_width=True, disabled=(st.session_state.faq_page == 1), key="faq_prev"):
                    st.session_state.faq_page -= 1
                    st.rerun()
            with pag_col3:
                st.markdown(f"""
                <div style="text-align:center;padding:0.5rem 0;color:#4a5568;font-size:0.9rem;">
                    第 <strong style="color:#1877f2;">{st.session_state.faq_page}</strong> / {total_pages} 页
                    &nbsp;&nbsp;共 <strong>{total_count}</strong> 条
                </div>
                """, unsafe_allow_html=True)
            with pag_col4:
                if st.button("下一页 ➡️", use_container_width=True, disabled=(st.session_state.faq_page == total_pages), key="faq_next"):
                    st.session_state.faq_page += 1
                    st.rerun()
            with pag_col5:
                if st.button("末页 ⏭️", use_container_width=True, disabled=(st.session_state.faq_page == total_pages), key="faq_last"):
                    st.session_state.faq_page = total_pages
                    st.rerun()


elif st.session_state.current_page == "智能问答":
    header_col1, header_col2 = st.columns([9, 1])
    with header_col1:
        st.markdown('<div class="qa-page-title">💬 智能问答</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown("<div style='text-align:right;font-size:1.25rem;'>🔔</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <h1 style="font-size:1.75rem;color:#1a202c;margin:0 0 0.5rem 0;">政务智能问答</h1>
        <p style="color:#718096;margin:0;font-size:0.95rem;">输入您想了解的政务问题，系统将基于政策文件、常见问题和工单记录为您解答</p>
    </div>
    """, unsafe_allow_html=True)

    if 'qa_history_loaded' not in st.session_state:
        st.session_state.qa_history_data = load_qa_history()
        st.session_state.qa_history_loaded = True

    history = st.session_state.qa_history_data

    if not history:
        st.markdown("<div style='text-align:center;color:#a0aec0;padding:3rem 0;font-size:0.9rem;'>暂无对话记录，开始您的第一次提问吧～</div>", unsafe_allow_html=True)
    else:
        last_date = None
        display_items = history[:-1] if history[-1].get("_streaming") else history
        for item in display_items:
            ts = datetime.fromisoformat(item['timestamp'])
            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M")
            if date_str != last_date:
                st.markdown(f"<div style='text-align:center;color:#a0aec0;font-size:0.75rem;margin:1rem 0 0.5rem 0;'>{date_str} {time_str}</div>", unsafe_allow_html=True)
                last_date = date_str
            else:
                st.markdown(f"<div style='text-align:center;color:#a0aec0;font-size:0.75rem;margin:1rem 0 0.5rem 0;'>{time_str}</div>", unsafe_allow_html=True)

            with st.chat_message("user", avatar="🧑"):
                st.markdown(item['question'])

            retrieval_html = f"""
            <div style="background:#f8fafc;border-radius:8px;padding:0.75rem 1rem;margin:0.5rem 0;border-left:3px solid #1877f2;">
                <div style="font-weight:600;color:#2d3748;margin-bottom:0.5rem;">🔍 检索过程</div>
                <div style="font-size:0.875rem;color:#4a5568;line-height:1.8;">
                    <div>• 政策库检索：找到 <strong>{item.get('policy_count', 0)}</strong> 条相关内容</div>
                    <div>• FAQ库检索：找到 <strong>{item.get('faq_count', 0)}</strong> 条相关内容</div>
                    <div>• 工单库检索：找到 <strong>{item.get('ticket_count', 0)}</strong> 条相关内容</div>
                    <div>• 重排序完成：精选 <strong>{item.get('rerank_count', 0)}</strong> 条最相关结果</div>
                </div>
            </div>
            """
            st.markdown(retrieval_html, unsafe_allow_html=True)

            strategy_map = {
                "high": ("高置信度", "#38a169", "#e6f7ed"),
                "medium": ("中等置信度", "#d69e2e", "#fff8e6"),
                "low": ("低置信度", "#e53e3e", "#fde8e8")
            }
            s_text, s_color, s_bg = strategy_map.get(item.get('strategy', 'high'), strategy_map['high'])

            with st.chat_message("assistant", avatar="🏛️"):
                st.markdown(f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:500;margin-right:0.5rem;background:{s_bg};color:{s_color};'>{s_text}</span>", unsafe_allow_html=True)
                st.markdown(item['answer'])

        # 处理正在流式中的最后一条
        if history and history[-1].get("_streaming"):
            last_item = history[-1]
            ts = datetime.fromisoformat(last_item['timestamp'])
            time_str = ts.strftime("%H:%M")
            st.markdown(f"<div style='text-align:center;color:#a0aec0;font-size:0.75rem;margin:1rem 0 0.5rem 0;'>{time_str}</div>", unsafe_allow_html=True)

            with st.chat_message("user", avatar="🧑"):
                st.markdown(last_item['question'])

            with st.chat_message("assistant", avatar="🏛️"):
                with st.spinner("正在检索相关信息..."):
                    import time
                    t0 = time.time()
                    stream_gen = rag_engine.query(last_item['question'], stream=True)
                    response_obj = next(stream_gen)
                    t_ret = round(time.time() - t0, 2)

                # 显示检索过程
                retrieval_html = f"""
                <div style="background:#f8fafc;border-radius:8px;padding:0.75rem 1rem;margin:0.5rem 0;border-left:3px solid #1877f2;">
                    <div style="font-weight:600;color:#2d3748;margin-bottom:0.5rem;">🔍 检索过程</div>
                    <div style="font-size:0.875rem;color:#4a5568;line-height:1.8;">
                        <div>• 政策库检索：找到 <strong>{response_obj.policy_count}</strong> 条相关内容</div>
                        <div>• FAQ库检索：找到 <strong>{response_obj.faq_count}</strong> 条相关内容</div>
                        <div>• 工单库检索：找到 <strong>{response_obj.ticket_count}</strong> 条相关内容</div>
                        <div>• 重排序完成：精选 <strong>{response_obj.rerank_count}</strong> 条最相关结果</div>
                    </div>
                </div>
                """
                st.markdown(retrieval_html, unsafe_allow_html=True)

                strategy_map = {
                    "high": ("高置信度", "#38a169", "#e6f7ed"),
                    "medium": ("中等置信度", "#d69e2e", "#fff8e6"),
                    "low": ("低置信度", "#e53e3e", "#fde8e8")
                }
                s_text, s_color, s_bg = strategy_map.get(response_obj.strategy, strategy_map['high'])
                st.markdown(f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:500;margin-right:0.5rem;background:{s_bg};color:{s_color};'>{s_text}</span>", unsafe_allow_html=True)

                # 流式输出答案
                answer_text = st.write_stream(stream_gen)

                # 保存完整记录
                elapsed = round(time.time() - t0, 2)
                last_item["answer"] = answer_text if isinstance(answer_text, str) else "".join(answer_text) if isinstance(answer_text, list) else str(answer_text)
                last_item["strategy"] = response_obj.strategy
                last_item["max_score"] = response_obj.max_score
                last_item["retrieval_info"] = response_obj.retrieval_info
                last_item["references"] = [
                    {"content": r.content, "source": r.source,
                     "section": r.section, "score": r.score, "kb_type": r.kb_type}
                    for r in response_obj.references
                ]
                last_item["response_time"] = elapsed
                last_item["policy_count"] = response_obj.policy_count
                last_item["faq_count"] = response_obj.faq_count
                last_item["ticket_count"] = response_obj.ticket_count
                last_item["rerank_count"] = response_obj.rerank_count
                last_item["_streaming"] = False
                last_item["category"] = classify_question(last_item['question'])

                save_qa_history(history)
                st.rerun()

    input_container = st.container()
    with input_container:
        with st.form(key="qa_form", clear_on_submit=True):
            question = st.text_area("问题输入", height=80, placeholder="请输入您的问题，例如：如何重置密码？", label_visibility="collapsed", key="qa_input")
            btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 3])
            with btn_col2:
                submit_btn = st.form_submit_button("立即提问", type="primary", use_container_width=True)

    if submit_btn and question.strip():
        st.session_state.query_count += 1
        import time
        start_time = time.time()

        q_text = question.strip()

        # 先将用户问题写入历史（用于立即显示）
        temp_record = {
            "timestamp": datetime.now().isoformat(),
            "question": q_text,
            "answer": "",
            "strategy": "high",
            "max_score": 0,
            "retrieval_info": {},
            "references": [],
            "response_time": 0,
            "category": classify_question(q_text),
            "policy_count": 0,
            "faq_count": 0,
            "ticket_count": 0,
            "rerank_count": 0,
            "_streaming": True
        }
        st.session_state.qa_history_data.append(temp_record)
        st.rerun()


elif st.session_state.current_page == "反馈记录":
    render_top_header("反馈记录")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">反馈明细</div>', unsafe_allow_html=True)
    if st.session_state.feedback_list:
        df_fb = pd.DataFrame(st.session_state.feedback_list)
        st.dataframe(df_fb, use_container_width=True, hide_index=True)
    else:
        st.info("暂无反馈记录")
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.current_page == "系统管理":
    render_top_header("系统管理")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">系统配置</div>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.text_input("Embedding 模型", value=config.EMBEDDING_MODEL_NAME, disabled=True)
        st.text_input("Reranker 模型", value=config.RERANKER_MODEL_NAME, disabled=True)
        st.text_input("大模型", value=config.DEEPSEEK_MODEL, disabled=True)
    with col_c2:
        st.text_input("检索 Top K", value=str(config.RETRIEVE_TOP_K), disabled=True)
        st.text_input("重排序 Top N", value=str(config.RERANK_TOP_N), disabled=True)
        st.text_input("API Key", value=config.DEEPSEEK_API_KEY[:8] + "****", disabled=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 一键构建所有知识库索引", type="primary"):
        with st.spinner("正在构建所有知识库索引..."):
            result = rag_engine.build_all_indexes()
        st.success("所有知识库索引构建完成！")
        for kb_name, count in result.items():
            st.write(f"- {kb_name}：{count} 条")

    st.markdown('</div>', unsafe_allow_html=True)
