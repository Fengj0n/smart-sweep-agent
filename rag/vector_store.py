import os
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = self._init_vector_store()
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def _init_vector_store(self):
        persist_directory = get_abs_path(chroma_conf["persist_directory"])
        try:
            return Chroma(
                collection_name=chroma_conf["collection_name"],
                embedding_function=embed_model,
                persist_directory=persist_directory,
            )
        except KeyError as e:
            if str(e).strip("'") == "_type":
                logger.warning(f"[向量库]检测到旧版 Chroma 配置文件损坏，正在重建：{persist_directory}")
                shutil.rmtree(persist_directory, ignore_errors=True)
                return Chroma(
                    collection_name=chroma_conf["collection_name"],
                    embedding_function=embed_model,
                    persist_directory=persist_directory,
                )
            raise

    def _is_vector_store_empty(self) -> bool:
        collection = getattr(self.vector_store, "_collection", None)
        if collection is None:
            return False

        try:
            return collection.count() == 0
        except Exception:
            return False

    def _add_documents_in_batches(self, documents: List[Document], batch_size: int = 10):
        for index in range(0, len(documents), batch_size):
            batch = documents[index:index + batch_size]
            self.vector_store.add_documents(batch)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """返回候选文档，供上层执行本地重排。"""
        return self.vector_store.similarity_search(query, k=k)

    @staticmethod
    def _infer_source(path: str) -> str:
        return Path(path).name

    @staticmethod
    def _enrich_documents(documents: List[Document], source_path: str) -> List[Document]:
        source_name = Path(source_path).name
        file_type = Path(source_path).suffix.lstrip(".").lower()
        enriched_docs: List[Document] = []
        total = len(documents)
        for index, doc in enumerate(documents, start=1):
            metadata = dict(doc.metadata or {})
            metadata.update(
                {
                    "source": source_name,
                    "source_path": source_path,
                    "file_type": file_type,
                    "chunk_index": index,
                    "chunk_total": total,
                }
            )
            enriched_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        return enriched_docs

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            md5_store = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_store):
                open(md5_store, "w", encoding="utf-8").close()
                return False

            with open(md5_store, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: List[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.warning(f"[加载知识库]{path}无法计算MD5，跳过")
                continue

            if check_md5_hex(md5_hex):
                if self._is_vector_store_empty():
                    logger.warning(f"[加载知识库]{path}的MD5已记录，但当前向量库为空，重新导入")
                else:
                    logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                    continue

            try:
                documents: List[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: List[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                enriched_documents = self._enrich_documents(split_document, path)
                self._add_documents_in_batches(enriched_documents, batch_size=10)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
