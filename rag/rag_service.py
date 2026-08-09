"""低 token 的 RAG 检索与总结服务。"""
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.model = chat_model
        self.chain = self.prompt_template | self.model | StrOutputParser()
        self.last_sources: List[Dict[str, str]] = []

    @staticmethod
    def _terms(text: str) -> List[str]:
        chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text or "")
        ascii_words = re.findall(r"[A-Za-z0-9_-]{2,}", (text or "").lower())
        return chinese + ascii_words

    def _rerank(self, query: str, docs: List[Document], limit: int = 3) -> List[Document]:
        terms = self._terms(query)
        unique: Dict[str, Document] = {}
        scored = []
        for index, doc in enumerate(docs):
            content = doc.page_content or ""
            source = str((doc.metadata or {}).get("source", "未知来源"))
            key = f"{source}:{content[:80]}"
            if key in unique:
                continue
            unique[key] = doc
            score = sum(2 for term in terms if term in content)
            score += sum(1 for term in terms if term in source)
            score -= index * 0.05
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def retriever_docs(self, query: str) -> List[Document]:
        docs = self.vector_store.similarity_search(query, k=6)
        return self._rerank(query, docs, limit=3)

    def _build_context(self, docs: List[Document], max_chars: int = 2200) -> str:
        lines = []
        total_chars = 0
        self.last_sources = []
        for counter, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            source = str(metadata.get("source", "未知来源"))
            chunk_index = str(metadata.get("chunk_index", "-"))
            content = (doc.page_content or "").strip()
            available = max_chars - total_chars
            if available <= 0:
                break
            content = content[:available]
            lines.append(f"[{counter}] {content}")
            self.last_sources.append({"source": source, "chunk": chunk_index})
            total_chars += len(content)
        return "\n".join(lines)

    @lru_cache(maxsize=64)
    def rag_summarize(self, query: str) -> str:
        query = (query or "").strip()[:180]
        if not query:
            return "请输入问题。"
        try:
            docs = self.retriever_docs(query)
            context = self._build_context(docs)
            if not context:
                return "知识库中暂时没有相关资料。"
            result = str(self.chain.invoke({"input": query, "context": context})).strip()
            answer = result[:1600].rstrip()
            sources = []
            for item in self.last_sources:
                source = item["source"]
                if source not in sources:
                    sources.append(source)
            if sources:
                answer += "\n\n来源：" + "、".join(sources[:2])
            return answer
        except Exception as exc:
            logger.error(f"[rag_summarize]执行失败：{exc}", exc_info=True)
            return "知识库查询暂时失败，请稍后重试。"


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
