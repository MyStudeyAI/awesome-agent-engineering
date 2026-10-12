"""
Lesson 06 — 进阶检索：混合检索 + Rerank
========================================
本脚本做四个实验，对比四种检索方式：
    ① 向量检索翻车现场（对编号/专有名词不敏感）
    ② BM25 关键词检索（精确匹配编号）
    ③ 混合检索（RRF 融合向量+BM25）
    ④ Rerank 重排序（召回+精排两段式）

运行：python rag-lessons/06_advanced_retrieval/code.py
"""
from __future__ import annotations

import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(model_dir)

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "embedding-3"
LOCAL_EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5" # 本地向量模型：中文效果好、体积小（约 100MB）、512 维
COLLECTION_NAME = "lesson06_docs"
CHROMA_PATH = "./chroma_db_06"

# 全局加载一次，避免每次调用都重新加载
_local_model = None

def get_local_model():
    global _local_model
    if _local_model is None:
        print(f"⏳ 正在加载本地向量模型 {LOCAL_EMBEDDING_MODEL_NAME} ...")
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
        print("✅ 本地向量模型加载完成")
    return _local_model

# ──────────────────────────────────────────────────────────────
# 文档集：故意混入带编号/专有名词的文档，制造向量检索的翻车场景。
# ──────────────────────────────────────────────────────────────
DOCUMENTS = [
    # 语义类（向量检索擅长）
    "员工入职满一年享有带薪年假福利，工龄越长假期越多。",
    "公司每年提供免费体检，覆盖所有正式员工。",
    "出差住宿费用可以申请报销，需要保留发票。",
    # 编号/专有名词类（BM25 擅长，向量检索容易漏）
    "表单 FORM-A12 用于申请年假，需在 OA 系统提交并附工号。",
    "故障码 ERR_0x42 表示网络连接超时，请检查防火墙配置。",
    "产品型号 PRJ-B200 的保修期为 24 个月，联系售后 8001。",
    "审批流 WF-CL-3 需经过部门主管和财务双重确认。",
    # 干扰项（语义沾边但不是答案）
    "年假天数根据职级确定，经理级及以上额外增加 3 天。",
]

# 实验用的查询：带编号，向量检索容易翻车
#QUERY = "表单 FORM-A12 怎么填？需要附什么信息？"
QUERY = "我想休个假放松一下，有什么政策？"
# 正确答案应该是 DOCUMENTS[3]

def create_zhipu_client() -> ZhipuAI:
    load_dotenv()
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("xxxx"):
        raise RuntimeError("请先在 .env 里配置 ZHIPUAI_API_KEY")
    return ZhipuAI(api_key=api_key)

def embed_texts(client: ZhipuAI, texts: list[str]) -> np.ndarray:
    # resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # sorted_data = sorted(resp.data, key=lambda x: x.index)
    # return np.array([d.embedding for d in sorted_data])

    # 用本地 sentence-transformers 模型把文本变成向量
        model = get_local_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def simple_tokenize(text: str) -> list[str]:
    """简易分词：英文按非字母数字切，中文按字切。

    真实场景会用 jieba 等专业分词器，这里用简易版保持依赖少。
    """
    import re

    # 英文/数字 token
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    # 中文字符（每个字作为一个 token）
    tokens += [c for c in text if "\u4e00" <= c <= "\u9fff"]
    return [t.lower() for t in tokens]

def build_chroma(doc_vectors: np.ndarray):
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = db.get_or_create_collection(name=COLLECTION_NAME)
    col.add(
        documents=DOCUMENTS,
        embeddings=doc_vectors.tolist(),
        ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
    )
    return col

def print_results(label: str, ranked_docs: list[tuple[str, float]], top_k: int = 5):
    """打印检索结果。ranked_docs: [(doc, score)]，按分数从高到低。"""
    print(f"\n【{label}】Top-{top_k}：")
    for i, (doc, score) in enumerate(ranked_docs[:top_k], 1):
        # 标记正确答案
        mark = " ✅正确" if doc == DOCUMENTS[3] else ""
        print(f"  {i}. 分数={score:.4f} | {doc[:40]}...{mark}")

# ════════════════════════════════════════════════════════════
# 实验①：向量检索（看它怎么翻车）
# ════════════════════════════════════════════════════════════
def vector_search(doc_vectors: np.ndarray, query_vec: np.ndarray) -> list[tuple]:
    """向量检索：算余弦相似度排序。返回 [(doc, score)]。"""
    sims = [(DOCUMENTS[i], cosine_sim(query_vec, dv)) for i, dv in enumerate(doc_vectors)]
    return sorted(sims, key=lambda x: x[1], reverse=True)

# ════════════════════════════════════════════════════════════
# 实验②：BM25 关键词检索
# ════════════════════════════════════════════════════════════
def bm25_search(query: str) -> list[tuple]:
    """BM25 关键词检索。返回 [(doc, score)]。"""
    tokenized_docs = [simple_tokenize(d) for d in DOCUMENTS]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(zip(DOCUMENTS, scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked]

# ════════════════════════════════════════════════════════════
# 实验③：混合检索（RRF 融合）
# ════════════════════════════════════════════════════════════
def rrf_fusion(
    vector_results: list[tuple], bm25_results: list[tuple], k: int = 60
) -> list[tuple]:
    """RRF (Reciprocal Rank Fusion) 融合两路检索结果。

    公式：RRF(doc) = Σ 1/(k + rank)
    一个文档如果在两路里都排名靠前，得分就高。
    """
    scores = {}
    # 向量检索的排名贡献
    for rank, (doc, _) in enumerate(vector_results, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    # BM25 的排名贡献
    for rank, (doc, _) in enumerate(bm25_results, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
# ════════════════════════════════════════════════════════════
# 实现加权融合（替代 RRF）
def weighted_fusion(
    vec_results: list[tuple], bm25_results: list[tuple], alpha=0.5
) -> list[tuple]:
    """加权融合。

    公式：R最终分数 = α × 向量分数 + (1-α) × BM25分数
    α 通常 0.5~0.7。
    查询偏关键词时调高 BM25 权重，偏语义时调高向量权重。
    """
    """alpha 是向量检索的权重，1-alpha 是 BM25 的权重。"""
    # 先把两路分数各自归一化到 0~1（因为量纲不同）
    def normalize(results):
        scores = np.array([s for _, s in results])
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return {d: s for (d, _), s in zip(results, scores)}
    vec_norm = normalize(vec_results)
    bm25_norm = normalize(bm25_results)
    all_docs = set(vec_norm) | set(bm25_norm)
    fused = {d: alpha * vec_norm.get(d, 0) + (1 - alpha) * bm25_norm.get(d, 0) for d in all_docs}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# ════════════════════════════════════════════════════════════
# 实验④：Rerank 重排序（召回+精排两段式）
# ════════════════════════════════════════════════════════════
def lightweight_rerank(query: str, candidate_docs: list[str]) -> list[tuple]:
    """轻量 Rerank：基于查询与文档的精确匹配度重排。

    ⚠️ 这是教学版简化实现，用"查询中关键 token 在文档中的命中率"来模拟
       cross-encoder 的精排效果。真实生产请换 bge-reranker / 智谱 reranker，
       流程完全一样，只是把这里的打分函数换成交叉编码器。
    """
    query_tokens = set(simple_tokenize(query))
    scored = []
    for doc in candidate_docs:
        doc_tokens = set(simple_tokenize(doc))
        # 查询 token 在文档中的命中率（精确匹配度）
        if query_tokens:
            hit_rate = len(query_tokens & doc_tokens) / len(query_tokens)
        else:
            hit_rate = 0
        scored.append((doc, hit_rate))
    return sorted(scored, key=lambda x: x[1], reverse=True)

def main():
    print("=" * 60)
    print("Lesson 06 — 进阶检索：混合检索 + Rerank")
    print("=" * 60)

    client = create_zhipu_client()
    print("\n正在向量化...")
    doc_vectors = embed_texts(client, DOCUMENTS)
    query_vec = embed_texts(client, [QUERY])[0]
    build_chroma(doc_vectors)
    print(f"✅ {len(DOCUMENTS)} 篇文档已就绪")

    print(f"\n🔎 查询：{QUERY}")
    print(f"   （正确答案是含 FORM-A12 的那条文档）")

    # ① 向量检索
    print("\n" + "═" * 60)
    print("① 向量检索（看它对编号 FORM-A12 敏不敏感）")
    print("═" * 60)
    vec_results = vector_search(doc_vectors, query_vec)
    print_results("向量检索", vec_results)
    print("\n👉 观察：正确答案可能没排到第一。向量检索对编号 FORM-A12 不敏感，")
    print("   它按'语义'找，可能召回'年假福利'这种语义沾边的文档。")

    # ② BM25
    print("\n" + "═" * 60)
    print("② BM25 关键词检索（看它怎么精准命中编号）")
    print("═" * 60)
    bm25_results = bm25_search(QUERY)
    print_results("BM25 检索", bm25_results)
    print("\n👉 观察：含 FORM-A12 的文档应该排到前面。BM25 对精确编号敏感。")

    # ③ 混合检索
    print("\n" + "═" * 60)
    print("③ 混合检索（RRF 融合向量+BM25）")
    print("═" * 60)
    hybrid_results = rrf_fusion(vec_results, bm25_results)
    print_results("混合检索 (RRF)", hybrid_results)
    print("\n👉 观察：两路都认可（语义+关键词都匹配）的文档排到前面。")
    print("   正确答案的排名应该比纯向量检索更靠前。")

    print("\n" + "═" * 60)
    print("③ 加权融合")
    print("═" * 60)
    weighted_results = weighted_fusion(vec_results, bm25_results)
    print_results("加权融合", weighted_results)
    print("\n👉 观察：RRF 和 加权融合。")

    # ④ Rerank
    print("\n" + "═" * 60)
    print("④ Rerank 重排序（对混合检索召回的 Top-5 精排）")
    print("═" * 60)
    # 先用混合检索召回 top-5，再用 rerank 精排
    candidates = [doc for doc, _ in hybrid_results[:5]]
    reranked = lightweight_rerank(QUERY, candidates)
    print_results("Rerank 后", reranked)
    print("\n👉 观察：精排后，含 FORM-A12 的文档应该稳稳排到第一。")
    print("   这就是召回+精排两段式：先粗筛（高召回），再精选（高精度）。")

    print("\n" + "=" * 60)
    print("完成！生产级 RAG 的标配：混合检索（召回）+ Rerank（精排）。")
    print("=" * 60)

if __name__ == "__main__":
    main()
