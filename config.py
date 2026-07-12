import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 尝试从 Streamlit secrets 读取（Cloud 部署和本地 secrets.toml 用）
try:
    import streamlit as st
    _st_secrets = st.secrets
except Exception:
    _st_secrets = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
FAQ_DIR = os.path.join(DATA_DIR, "faq")
TICKETS_DIR = os.path.join(DATA_DIR, "tickets")

VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# 从 Streamlit secrets 或环境变量读取 API Key
# 优先级：Streamlit secrets > 环境变量 / .env 文件（已通过 load_dotenv 加载）
DEEPSEEK_API_KEY = (
    _st_secrets.get("DEEPSEEK_API_KEY", "")
    or os.getenv("DEEPSEEK_API_KEY", "")
)

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "未找到 DEEPSEEK_API_KEY！请通过以下任一方式配置：\n"
        "  1. 在 .streamlit/secrets.toml 中添加 DEEPSEEK_API_KEY = \"your-key\"（推荐）\n"
        "  2. 设置环境变量：export DEEPSEEK_API_KEY=your-key\n"
        "  3. 在项目根目录创建 .env 文件：DEEPSEEK_API_KEY=your-key"
    )

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

RETRIEVE_TOP_K = 3
RERANK_TOP_N = 3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

MAX_TOKENS = 512
TEMPERATURE = 0.3

SUPPORTED_DOC_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"]

# 是否启用 PDF 图片 OCR 识别（默认启用，使用 RapidOCR 基于 ONNX Runtime）
ENABLE_OCR = True
OCR_LANG = "ch"  # 保留兼容字段（RapidOCR 自动支持中英文）

# 相似度阈值（向量检索阶段，Chroma 返回的是余弦相似度，范围 0~1）
# 优化：从 0.6 降低到 0.45，让更多"可能相关"的结果进入候选池，再由 Rerank 精筛
SIMILARITY_THRESHOLD = 0.45

# Rerank 分数阈值（BGE-Reranker 输出的原始 logits，可正可负）
RERANK_THRESHOLD = 0.3

# 高置信度阈值（>= 此值直接回答，不加提示）
HIGH_CONFIDENCE_THRESHOLD = 0.7

# ========== 混合检索配置 ==========
# 是否启用混合检索（向量检索 + BM25 关键词检索）
ENABLE_HYBRID_SEARCH = True

# BM25 检索权重（0.0~1.0，越大越偏向关键词匹配）
BM25_WEIGHT = 0.4

# 向量检索权重（0.0~1.0，越大越偏向语义匹配）
VECTOR_WEIGHT = 0.6

# 是否启用同义词扩展
ENABLE_SYNONYM_EXPANSION = True

# 是否启用文本预处理（分词、去停用词）
ENABLE_TEXT_PREPROCESSING = True

# BM25 检索召回数量（每个知识库）
BM25_TOP_K = 5

# ========== 政务场景同义词词典 ==========
# 每个词组内的所有词视为同义，检索时互相扩展
SYNONYM_DICT = {
    "过户": ["赠与", "转移", "更名", "变更", "转让", "移交"],
    "房子": ["房屋", "房产", "不动产", "住宅", "楼房", "物业"],
    "孩子": ["子女", "儿子", "女儿", "未成年", "小孩"],
    "税": ["税费", "契税", "税款", "税收", "费用", "印花税"],
    "办理": ["申请", "申报", "登记", "注册", "受理", "审批"],
    "身份证": ["身份证明", "证件", "ID", "居民身份证"],
    "户口": ["户籍", "户口本", "户籍证明", "常住人口登记"],
    "社保": ["社会保险", "医保", "养老保险", "工伤保险", "失业保险"],
    "公积金": ["住房公积金", "公积金账户", "缴存"],
    "营业执照": ["工商执照", "经营许可", "企业执照", "登记证书"],
    "结婚": ["婚姻", "结婚证", "婚姻登记", "配偶"],
    "离婚": ["解除婚姻", "离婚证", "婚姻终止", "分手"],
    "出生": ["出生证", "出生证明", "出生医学证明", "出生登记"],
    "死亡": ["去世", "逝世", "死亡证明", "注销户口"],
    "迁移": ["迁入", "迁出", "搬迁", "户口迁移", "迁居"],
    "丢失": ["遗失", "丢失", "补办", "补发", "挂失"],
    "查询": ["查看", "咨询", "了解", "询问", "查找"],
    "材料": ["资料", "文件", "证件", "证明", "手续"],
    "流程": ["程序", "步骤", "办法", "方式", "方法", "如何"],
    "时间": ["期限", "时长", "多久", "工作日", "天数"],
    "地点": ["地址", "位置", "哪里", "何处", "窗口"],
    "费用": ["收费", "价格", "多少钱", "金额", "花费"],
    "条件": ["要求", "资格", "标准", "前提", "限制"],
    "变更": ["修改", "更改", "更新", "调整", "变动"],
    "企业": ["公司", "单位", "机构", "组织", "商户"],
    "学校": ["院校", "教育机构", "幼儿园", "小学", "中学"],
    "医院": ["医疗机构", "诊所", "卫生院", "门诊"],
    "交通": ["出行", "公交", "地铁", "车辆", "驾驶"],
    "不动产": ["房产", "房屋", "土地", "物业", "房地产"],
}

# 中文停用词表（检索预处理时过滤掉的无意义词）
STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "它", "他", "她", "吗", "什么", "怎么", "哪", "哪些",
    "可以", "能", "能够", "需要", "应该", "必须", "可能", "大概", "也许",
    "请问", "请", "帮", "帮忙", "告诉", "知道", "想", "希望", "想要",
    "现在", "目前", "已经", "之前", "之后", "以后", "以前", "当时",
    "如果", "因为", "所以", "但是", "而且", "或者", "还是", "以及",
    "这个", "那个", "这些", "那些", "这样", "那样", "怎么", "怎样",
    "多少", "几个", "一些", "一点", "一下", "一直", "比较", "非常",
}

# 人工客服信息
CUSTOMER_SERVICE_PHONE = "400-123-4567"
CUSTOMER_SERVICE_HOURS = "工作日 9:00-18:00"

# 问答历史记录配置
QA_HISTORY_FILE = os.path.join(DATA_DIR, "qa_history.json")
QA_HISTORY_EXPIRE_HOURS = 24  # 历史记录过期时间（小时）


