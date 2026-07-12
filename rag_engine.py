import os
import csv
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_community.document_loaders import Docx2txtLoader
    _has_docx_loader = True
except ImportError:
    _has_docx_loader = False

from sentence_transformers import CrossEncoder

# jieba 分词（用于中文分词和 BM25 检索）
try:
    import jieba
    import jieba.analyse
    _has_jieba = True
except ImportError:
    _has_jieba = False

# BM25 检索
try:
    from rank_bm25 import BM25Okapi
    _has_bm25 = True
except ImportError:
    _has_bm25 = False

import config


@dataclass
class RetrievalResult:
    content: str
    source: str
    section: str
    score: float
    kb_type: str


@dataclass
class RAGResponse:
    answer: str
    retrieval_info: Dict[str, int]
    references: List[RetrievalResult]
    max_score: float = 0.0
    strategy: str = "high"  # high / medium / low
    policy_count: int = 0
    faq_count: int = 0
    ticket_count: int = 0
    rerank_count: int = 0


class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.reranker = None
        self.llm = None
        self.vector_stores = {
            "policy": None,
            "faq": None,
            "ticket": None
        }
        # BM25 索引：{kb_type: (bm25_instance, doc_list)}
        self.bm25_indexes = {
            "policy": None,
            "faq": None,
            "ticket": None
        }
        self._initialized = False
        self._ocr_reader = None
        self._ocr_available = None

    def initialize(self) -> None:
        if self._initialized:
            return

        print("正在加载 Embedding 模型...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        print("正在加载 Reranker 模型...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

        print("正在初始化大模型...")
        self.llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE
        )

        if config.ENABLE_OCR:
            print("正在预加载 RapidOCR 模型...")
            self._init_ocr()

        self._load_vector_stores()
        self._initialized = True
        print("RAG 引擎初始化完成")

    def _load_vector_stores(self) -> None:
        for kb_type in ["policy", "faq", "ticket"]:
            persist_dir = os.path.join(config.VECTOR_STORE_DIR, kb_type)
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                self.vector_stores[kb_type] = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=f"{kb_type}_collection"
                )
                print(f"  - 已加载 {kb_type} 知识库向量索引")
            else:
                print(f"  - {kb_type} 知识库向量索引不存在，需要构建")

        # 加载向量库后构建 BM25 索引
        if config.ENABLE_HYBRID_SEARCH and _has_bm25:
            print("正在构建 BM25 关键词索引...")
            for kb_type in ["policy", "faq", "ticket"]:
                self._build_bm25_index(kb_type)

    def _build_bm25_index(self, kb_type: str) -> None:
        """为指定知识库构建 BM25 索引"""
        if not _has_bm25 or not _has_jieba:
            print(f"  - {kb_type} BM25 索引构建跳过（未安装 rank_bm25 或 jieba）")
            return

        vs = self.vector_stores[kb_type]
        if vs is None:
            return

        try:
            # 从 Chroma 中获取所有文档
            all_data = vs.get(include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            metas = all_data.get("metadatas", [])

            if not docs:
                return

            # 对每个文档分词
            tokenized_docs = []
            for text in docs:
                tokens = self._tokenize(text)
                tokenized_docs.append(tokens)

            bm25 = BM25Okapi(tokenized_docs)
            # 保存 BM25 实例和对应的文档内容列表
            self.bm25_indexes[kb_type] = (bm25, docs, metas)
            print(f"  - {kb_type} BM25 索引构建完成（{len(docs)} 篇文档）")
        except Exception as e:
            print(f"  - {kb_type} BM25 索引构建失败: {e}")
            self.bm25_indexes[kb_type] = None

    def _tokenize(self, text: str) -> List[str]:
        """中文分词，去除停用词"""
        if not text:
            return []

        if _has_jieba and config.ENABLE_TEXT_PREPROCESSING:
            tokens = list(jieba.cut(text))
            # 去除停用词和单字符
            tokens = [t.strip() for t in tokens
                      if t.strip() and t not in config.STOP_WORDS and len(t.strip()) > 1]
            return tokens if tokens else list(jieba.cut(text))
        else:
            # 简单按空格和标点切分
            return [t for t in re.split(r'[\s，。！？；,\.!?;]+', text) if t.strip()]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（用于实体识别）"""
        if _has_jieba:
            keywords = jieba.analyse.extract_tags(text, topK=10)
            return keywords
        return self._tokenize(text)

    def _expand_query_with_synonyms(self, query: str) -> str:
        """同义词扩展：将用户查询中的词替换为同义词集合"""
        if not config.ENABLE_SYNONYM_EXPANSION:
            return query

        expanded_terms = [query]

        # 遍历同义词词典，查找查询中是否包含某个词
        for key, synonyms in config.SYNONYM_DICT.items():
            if key in query:
                # 添加所有同义词
                for syn in synonyms:
                    if syn not in query:
                        expanded_terms.append(syn)
            else:
                # 检查查询中是否包含同义词词典中的某个同义词
                for syn in synonyms:
                    if syn in query:
                        expanded_terms.append(key)
                        # 添加其他同义词
                        for other_syn in synonyms:
                            if other_syn != syn and other_syn not in query:
                                expanded_terms.append(other_syn)
                        break

        expanded_query = " ".join(expanded_terms)
        if expanded_query != query:
            print(f"[同义词扩展] 原查询: {query}")
            print(f"[同义词扩展] 扩展后: {expanded_query}")
        return expanded_query

    def _init_ocr(self) -> None:
        if self._ocr_available is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_reader = RapidOCR()
            self._ocr_available = True
            print("  - RapidOCR 模型加载完成")
        except Exception as e:
            self._ocr_available = False
            self._ocr_reader = None
            print(f"  - RapidOCR 加载失败，将跳过图片OCR：{e}")

    def _ocr_image(self, image_bytes: bytes) -> str:
        if not self._ocr_available or self._ocr_reader is None:
            return ""
        try:
            import io
            import numpy as np
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(img)
            # RapidOCR 返回 (result, elapse)，result = [[box, text, score], ...]
            result, elapse = self._ocr_reader(img_array)
            texts = []
            if result:
                for line in result:
                    texts.append(line[1])
            text = "\n".join(texts)
            return text.strip()
        except Exception as e:
            print(f"OCR 识别失败: {e}")
            return ""

    def _extract_pdf_images_ocr(self, file_path: str) -> Dict[int, str]:
        if not self._ocr_available:
            return {}
        try:
            import fitz
            doc = fitz.open(file_path)
            page_ocr_text = {}
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                if not image_list:
                    continue
                page_texts = []
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ocr_text = self._ocr_image(image_bytes)
                    if ocr_text:
                        page_texts.append(ocr_text)
                if page_texts:
                    page_ocr_text[page_num] = "\n".join(page_texts)
            doc.close()
            return page_ocr_text
        except Exception as e:
            print(f"提取PDF图片OCR失败 {file_path}: {e}")
            return {}

    def _render_pdf_page_ocr(self, file_path: str) -> Dict[int, str]:
        """对扫描版 PDF，将每页渲染为图片后 OCR 识别"""
        if not self._ocr_available:
            return {}
        try:
            import fitz
            import io
            import numpy as np
            from PIL import Image

            doc = fitz.open(file_path)
            page_ocr_text = {}
            for page_num in range(len(doc)):
                page = doc[page_num]
                # 将页面渲染为高分辨率图片
                mat = fitz.Matrix(2.0, 2.0)  # 2倍缩放，提高清晰度
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
                ocr_text = self._ocr_image(image_bytes)
                if ocr_text:
                    page_ocr_text[page_num] = ocr_text
                    print(f"    第{page_num + 1}页 OCR 识别: {len(ocr_text)} 字符")
                else:
                    print(f"    ⚠️ 第{page_num + 1}页 OCR 识别结果为空")
            doc.close()
            return page_ocr_text
        except Exception as e:
            print(f"整页 OCR 渲染失败 {file_path}: {e}")
            return {}

    def _load_single_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        documents = []

        try:
            if ext == '.pdf':
                # 使用 pdfplumber 解析 PDF（比 PyPDFLoader 更稳定）
                import pdfplumber
                documents = []
                empty_pages = []
                all_empty = False

                with pdfplumber.open(file_path) as pdf:
                    total_pages = len(pdf.pages)
                    for page_idx, page in enumerate(pdf.pages):
                        text = (page.extract_text() or "").strip()
                        if text:
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "section": f"第{page_idx + 1}页",
                                    "kb_type": "policy",
                                    "page": page_idx
                                }
                            )
                            documents.append(doc)
                        else:
                            empty_pages.append(page_idx + 1)

                if empty_pages:
                    print(f"  ⚠️ {filename}：第 {empty_pages} 页未提取到文字")

                # 如果所有页面都没有文字，说明是扫描版 PDF，尝试整页 OCR
                if not documents and total_pages > 0:
                    all_empty = True
                    print(f"  🔍 {filename}：所有页面均为空，判定为扫描版 PDF，启动整页 OCR...")
                    if config.ENABLE_OCR and self._ocr_available:
                        self._init_ocr()
                        page_ocr_text = self._render_pdf_page_ocr(file_path)
                        for page_idx, ocr_text in page_ocr_text.items():
                            doc = Document(
                                page_content=ocr_text,
                                metadata={
                                    "source": filename,
                                    "section": f"第{page_idx + 1}页（含图片识别内容）",
                                    "kb_type": "policy",
                                    "page": page_idx
                                }
                            )
                            documents.append(doc)
                        if documents:
                            print(f"  ✅ {filename}：OCR 成功识别 {len(documents)} 页内容")
                        else:
                            print(f"  ❌ {filename}：OCR 也未能识别任何内容，该文件可能无法解析")
                    else:
                        print(f"  ❌ {filename}：OCR 未启用，无法解析扫描版 PDF")

                # 如果有部分页面有文字，再检查嵌入图片的 OCR
                elif documents and config.ENABLE_OCR and self._ocr_available:
                    self._init_ocr()
                    page_ocr_text = self._extract_pdf_images_ocr(file_path)
                    if page_ocr_text:
                        has_ocr = False
                        for doc in documents:
                            page_num = doc.metadata.get('page', 0)
                            if page_num in page_ocr_text:
                                ocr_text = page_ocr_text[page_num]
                                doc.page_content = doc.page_content + "\n\n[图片识别内容]\n" + ocr_text
                                doc.metadata["section"] = doc.metadata["section"] + "（含图片识别内容）"
                                has_ocr = True
                        if has_ocr:
                            print(f"  - {filename}：{len(page_ocr_text)} 页包含图片OCR内容")

            elif ext in ['.docx', '.doc']:
                if not _has_docx_loader:
                    print(f"跳过 {filename}：未安装 python-docx，无法读取 Word 文档")
                    return []
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                for i, doc in enumerate(docs):
                    doc.metadata["source"] = filename
                    doc.metadata["section"] = f"第{i + 1}段"
                    doc.metadata["kb_type"] = "policy"
                documents.extend(docs)

            elif ext in ['.txt', '.md', '.markdown']:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["section"] = "全文"
                    doc.metadata["kb_type"] = "policy"
                documents.extend(docs)

            else:
                print(f"跳过 {filename}：不支持的文件格式 {ext}")

        except Exception as e:
            print(f"加载 {filename} 失败: {e}")

        return documents

    def build_policy_index(self) -> int:
        if not os.path.exists(config.POLICIES_DIR):
            return 0

        documents = []
        all_files = []
        for ext in config.SUPPORTED_DOC_EXTENSIONS:
            all_files.extend([
                f for f in os.listdir(config.POLICIES_DIR)
                if f.lower().endswith(ext)
            ])
        all_files = list(set(all_files))

        for filename in all_files:
            file_path = os.path.join(config.POLICIES_DIR, filename)
            docs = self._load_single_document(file_path)
            documents.extend(docs)

        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        persist_dir = os.path.join(config.VECTOR_STORE_DIR, "policy")
        self.vector_stores["policy"] = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name="policy_collection"
        )
        # 重建 BM25 索引
        if config.ENABLE_HYBRID_SEARCH and _has_bm25:
            self._build_bm25_index("policy")
        return len(splits)

    def build_csv_index(self, kb_type: str, dir_path: str) -> int:
        if not os.path.exists(dir_path):
            return 0

        documents = []
        csv_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(dir_path, csv_file)
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    question = row.get('question', '')
                    answer = row.get('answer', '')
                    content = f"问题：{question}\n答案：{answer}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": csv_file,
                            "section": f"第{i + 1}条",
                            "kb_type": kb_type,
                            "question": question,
                            "answer": answer
                        }
                    )
                    documents.append(doc)

        if not documents:
            return 0

        persist_dir = os.path.join(config.VECTOR_STORE_DIR, kb_type)
        self.vector_stores[kb_type] = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name=f"{kb_type}_collection"
        )
        # 重建 BM25 索引
        if config.ENABLE_HYBRID_SEARCH and _has_bm25:
            self._build_bm25_index(kb_type)
        return len(documents)

    def build_all_indexes(self) -> Dict[str, int]:
        result = {}
        result["policy"] = self.build_policy_index()
        result["faq"] = self.build_csv_index("faq", config.FAQ_DIR)
        result["ticket"] = self.build_csv_index("ticket", config.TICKETS_DIR)
        return result

    def _retrieve_kb(self, kb_type: str, query: str, top_k: int) -> Tuple[List[RetrievalResult], int]:
        """检索单个知识库，返回 (过滤后结果, 过滤前总数)
        支持混合检索：向量检索（语义） + BM25 检索（关键词）
        """
        if self.vector_stores[kb_type] is None:
            return [], 0

        # ========== 同义词扩展 ==========
        expanded_query = self._expand_query_with_synonyms(query)

        # ========== 1. 向量检索（语义匹配） ==========
        # 用扩展后的查询进行向量检索
        vec_results = self.vector_stores[kb_type].similarity_search_with_score(expanded_query, k=top_k)
        total_before_filter = len(vec_results)

        # 用 content 作为去重键，合并向量检索和 BM25 检索的结果
        merged = {}  # {content_key: RetrievalResult}

        for doc, score in vec_results:
            distance = float(score)
            similarity = 1.0 / (1.0 + distance)

            # 低于阈值直接丢弃
            if similarity < config.SIMILARITY_THRESHOLD:
                continue

            content_key = doc.page_content[:100]
            merged[content_key] = RetrievalResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "未知"),
                section=doc.metadata.get("section", "未知"),
                score=similarity * config.VECTOR_WEIGHT,  # 加权
                kb_type=kb_type
            )

        # ========== 2. BM25 检索（关键词匹配） ==========
        if config.ENABLE_HYBRID_SEARCH and self.bm25_indexes[kb_type] is not None:
            bm25_results = self._bm25_search(kb_type, expanded_query, config.BM25_TOP_K)

            for r in bm25_results:
                content_key = r.content[:100]
                if content_key in merged:
                    # 已存在，累加分数（加权融合）
                    merged[content_key].score += r.score * config.BM25_WEIGHT
                else:
                    # 新增结果
                    r.score = r.score * config.BM25_WEIGHT
                    merged[content_key] = r

        retrieval_results = list(merged.values())
        return retrieval_results, total_before_filter

    def _bm25_search(self, kb_type: str, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25 关键词检索"""
        bm25_data = self.bm25_indexes[kb_type]
        if bm25_data is None:
            return []

        bm25, docs, metas = bm25_data
        if not docs:
            return []

        try:
            # 对查询分词
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # 获取 BM25 分数
            scores = bm25.get_scores(query_tokens)

            # 按分数降序排序，取 top_k
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            results = []
            for idx in ranked_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue

                meta = metas[idx] if idx < len(metas) else {}
                results.append(RetrievalResult(
                    content=docs[idx],
                    source=meta.get("source", "未知"),
                    section=meta.get("section", "未知"),
                    score=score,
                    kb_type=kb_type
                ))

            return results
        except Exception as e:
            print(f"[BM25 检索失败] {kb_type}: {e}")
            return []

    def _rerank(self, query: str, results: List[RetrievalResult], top_n: int) -> List[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self.reranker.predict(pairs)

        for i, r in enumerate(results):
            r.score = float(scores[i])

        # 先按分数降序排序
        ranked = sorted(results, key=lambda x: x.score, reverse=True)

        # 【新增】Rerank 分数阈值过滤：低于阈值的视为不相关，直接丢弃
        filtered = [r for r in ranked if r.score >= config.RERANK_THRESHOLD]

        return filtered[:top_n]

    def _generate_answer(self, query: str, references: List[RetrievalResult], stream: bool = False):
        if not references:
            if stream:
                yield "抱歉，未找到相关信息，请尝试其他问题。"
                return
            return "抱歉，未找到相关信息，请尝试其他问题。"

        context_parts = []
        for i, ref in enumerate(references, 1):
            context_parts.append(f"[{i}] 来源：{ref.source} - {ref.section}\n内容：{ref.content}")

        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{query}

请基于参考资料中的信息，给出准确、简洁的回答。如果参考资料中没有相关信息，请明确说明。回答时请使用中文。""")

        chain = prompt | self.llm | StrOutputParser()

        if stream:
            for chunk in chain.stream({"context": context, "query": query, "max_tokens": config.MAX_TOKENS}):
                yield chunk
        else:
            answer = chain.invoke({"context": context, "query": query, "max_tokens": config.MAX_TOKENS})
            return answer

    def query(self, question: str, stream: bool = False):
        import time
        t_total_start = time.time()

        if not self._initialized:
            t_init_start = time.time()
            self.initialize()
            print(f"[耗时] 引擎初始化: {time.time() - t_init_start:.2f}s")

        retrieval_info = {}
        all_results = []
        total_before_filter = 0

        kb_names = {
            "policy": "政策库",
            "faq": "FAQ库",
            "ticket": "工单库"
        }

        # ========== 第一阶段：向量检索 + 相似度阈值过滤 ==========
        t_retrieve_start = time.time()
        for kb_type, kb_name in kb_names.items():
            results, raw_count = self._retrieve_kb(kb_type, question, config.RETRIEVE_TOP_K)
            total_before_filter += raw_count
            retrieval_info[kb_name] = len(results)
            all_results.extend(results)

        t_retrieve = time.time() - t_retrieve_start
        print(f"[耗时] 检索阶段: {t_retrieve:.2f}s (召回 {total_before_filter} 条，过滤后 {len(all_results)} 条)")

        # ========== 第二阶段：Rerank 重排序 + 分数阈值过滤 ==========
        t_rerank_start = time.time()
        reranked = self._rerank(question, all_results, config.RERANK_TOP_N)
        t_rerank = time.time() - t_rerank_start

        if reranked:
            max_score = reranked[0].score
        else:
            max_score = 0.0

        print(f"[耗时] Rerank阶段: {t_rerank:.2f}s (过滤后 {len(reranked)} 条，最高分 {max_score:.4f})")

        # ========== 第三阶段：两层去重（按文档名 + 按文本相似度） ==========
        if reranked:
            # 第一层：按文档名去重，相同文档名只保留分数最高的那条
            seen_sources = {}
            dedup_by_source = []
            for r in reranked:
                source_key = r.source
                if source_key not in seen_sources:
                    seen_sources[source_key] = r
                    dedup_by_source.append(r)
            print(f"[检索日志] 第一层去重（按文档名）：从 {len(reranked)} 条减少到 {len(dedup_by_source)} 条")

            # 第二层：按前50个字符的文本相似度去重，相似度 > 0.8 的只保留分数最高的那条
            dedup_final = []
            for r in dedup_by_source:
                text_prefix = r.content[:50].strip()
                is_duplicate = False
                for kept in dedup_final:
                    kept_prefix = kept.content[:50].strip()
                    similarity = SequenceMatcher(None, text_prefix, kept_prefix).ratio()
                    if similarity > 0.8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    dedup_final.append(r)
            print(f"[检索日志] 第二层去重（按文本相似度）：从 {len(dedup_by_source)} 条减少到 {len(dedup_final)} 条")

            reranked = dedup_final
            if reranked:
                max_score = reranked[0].score

        # ========== 第四阶段：三段式兜底策略 ==========
        if not reranked or max_score < config.RERANK_THRESHOLD:
            strategy = "low"
            answer = (
                "抱歉，我的知识库中没有找到与您问题直接相关的信息。"
                "建议您换一种问法重新提问，或联系人工客服咨询。\n\n"
                f"📞 客服电话：{config.CUSTOMER_SERVICE_PHONE}\n"
                f"🕐 服务时间：{config.CUSTOMER_SERVICE_HOURS}"
            )
            print(f"[策略日志] 采用低置信度兜底策略（max_score={max_score:.4f} < {config.RERANK_THRESHOLD}）")
            t_total = time.time() - t_total_start
            print(f"[耗时] 总耗时: {t_total:.2f}s (检索{t_retrieve:.2f}s + Rerank{t_rerank:.2f}s)")

            response = RAGResponse(
                answer=answer,
                retrieval_info=retrieval_info,
                references=reranked,
                max_score=max_score,
                strategy=strategy,
                policy_count=retrieval_info.get("政策库", 0),
                faq_count=retrieval_info.get("FAQ库", 0),
                ticket_count=retrieval_info.get("工单库", 0),
                rerank_count=len(reranked)
            )
            if stream:
                yield response
                for ch in answer:
                    yield ch
                return
            return response

        elif max_score < config.HIGH_CONFIDENCE_THRESHOLD:
            strategy = "medium"
            print(f"[策略日志] 采用中等置信度策略（{config.RERANK_THRESHOLD} <= max_score={max_score:.4f} < {config.HIGH_CONFIDENCE_THRESHOLD}）")

            response_obj = RAGResponse(
                answer="",
                retrieval_info=retrieval_info,
                references=reranked,
                max_score=max_score,
                strategy=strategy,
                policy_count=retrieval_info.get("政策库", 0),
                faq_count=retrieval_info.get("FAQ库", 0),
                ticket_count=retrieval_info.get("工单库", 0),
                rerank_count=len(reranked)
            )

            t_llm_start = time.time()
            if stream:
                yield response_obj
                full_answer = []
                for chunk in self._generate_answer(question, reranked, stream=True):
                    full_answer.append(chunk)
                    yield chunk
                suffix = "\n\n---\n⚠️ 以上信息仅供参考，建议以官方文件为准。"
                yield suffix
                t_llm = time.time() - t_llm_start
                t_total = time.time() - t_total_start
                print(f"[耗时] LLM生成: {t_llm:.2f}s")
                print(f"[耗时] 总耗时: {t_total:.2f}s (检索{t_retrieve:.2f}s + Rerank{t_rerank:.2f}s + LLM{t_llm:.2f}s)")
                return
            else:
                answer = self._generate_answer(question, reranked)
                answer += "\n\n---\n⚠️ 以上信息仅供参考，建议以官方文件为准。"
                t_llm = time.time() - t_llm_start

        else:
            strategy = "high"
            print(f"[策略日志] 采用高置信度策略（max_score={max_score:.4f} >= {config.HIGH_CONFIDENCE_THRESHOLD}）")

            response_obj = RAGResponse(
                answer="",
                retrieval_info=retrieval_info,
                references=reranked,
                max_score=max_score,
                strategy=strategy,
                policy_count=retrieval_info.get("政策库", 0),
                faq_count=retrieval_info.get("FAQ库", 0),
                ticket_count=retrieval_info.get("工单库", 0),
                rerank_count=len(reranked)
            )

            t_llm_start = time.time()
            if stream:
                yield response_obj
                for chunk in self._generate_answer(question, reranked, stream=True):
                    yield chunk
                t_llm = time.time() - t_llm_start
                t_total = time.time() - t_total_start
                print(f"[耗时] LLM生成: {t_llm:.2f}s")
                print(f"[耗时] 总耗时: {t_total:.2f}s (检索{t_retrieve:.2f}s + Rerank{t_rerank:.2f}s + LLM{t_llm:.2f}s)")
                return
            else:
                answer = self._generate_answer(question, reranked)
                t_llm = time.time() - t_llm_start

        t_total = time.time() - t_total_start
        print(f"[耗时] LLM生成: {t_llm:.2f}s")
        print(f"[耗时] 总耗时: {t_total:.2f}s (检索{t_retrieve:.2f}s + Rerank{t_rerank:.2f}s + LLM{t_llm:.2f}s)")

        return RAGResponse(
            answer=answer,
            retrieval_info=retrieval_info,
            references=reranked,
            max_score=max_score,
            strategy=strategy,
            policy_count=retrieval_info.get("政策库", 0),
            faq_count=retrieval_info.get("FAQ库", 0),
            ticket_count=retrieval_info.get("工单库", 0),
            rerank_count=len(reranked)
        )

    def get_kb_stats(self) -> Dict[str, int]:
        stats = {}
        kb_names = {
            "policy": "政策库",
            "faq": "FAQ库",
            "ticket": "工单库"
        }
        for kb_type, kb_name in kb_names.items():
            if self.vector_stores[kb_type] is not None:
                stats[kb_name] = self.vector_stores[kb_type]._collection.count()
            else:
                stats[kb_name] = 0
        return stats

    def is_initialized(self) -> bool:
        return self._initialized

    def get_files_by_kb(self, kb_type: str) -> List[Dict[str, str]]:
        kb_dir_map = {
            "policy": (config.POLICIES_DIR, config.SUPPORTED_DOC_EXTENSIONS),
            "faq": (config.FAQ_DIR, [".csv"]),
            "ticket": (config.TICKETS_DIR, [".csv"])
        }
        dir_path, exts = kb_dir_map.get(kb_type, (None, []))
        if not dir_path or not os.path.exists(dir_path):
            return []

        files = []
        for ext in exts:
            files.extend([f for f in os.listdir(dir_path) if f.lower().endswith(ext)])
        files = sorted(list(set(files)))

        result = []
        for f in files:
            file_path = os.path.join(dir_path, f)
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            result.append({
                "name": f,
                "path": file_path,
                "size": size,
                "update_time": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        return result

    def save_uploaded_file(self, uploaded_file, target_dir: str) -> str:
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def save_uploaded_files_to_kb(self, uploaded_files, kb_type: str) -> List[str]:
        kb_dir_map = {
            "policy": config.POLICIES_DIR,
            "faq": config.FAQ_DIR,
            "ticket": config.TICKETS_DIR
        }
        target_dir = kb_dir_map.get(kb_type)
        if not target_dir:
            return []
        os.makedirs(target_dir, exist_ok=True)
        saved = []
        for uf in uploaded_files:
            file_path = os.path.join(target_dir, uf.name)
            with open(file_path, 'wb') as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        return saved

    def delete_file_from_kb(self, kb_type: str, filename: str) -> bool:
        kb_dir_map = {
            "policy": config.POLICIES_DIR,
            "faq": config.FAQ_DIR,
            "ticket": config.TICKETS_DIR
        }
        target_dir = kb_dir_map.get(kb_type)
        if not target_dir:
            return False
        file_path = os.path.join(target_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                print(f"删除文件 {filename} 失败: {e}")
                return False
        return False

    def get_kb_dir(self, kb_type: str) -> str:
        kb_dir_map = {
            "policy": config.POLICIES_DIR,
            "faq": config.FAQ_DIR,
            "ticket": config.TICKETS_DIR
        }
        return kb_dir_map.get(kb_type, "")
