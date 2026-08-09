"""文件处理工具。"""

import os
import hashlib
from typing import Tuple, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ✅ 使用相对导入
from utils.logger_handler import logger


def get_file_md5_hex(filepath: str):
    """
    计算文件的 MD5 值。

    Args:
        filepath: 文件路径

    Returns:
        MD5 十六进制字符串，失败返回 None
    """
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096  # 4KB分片，避免文件过大爆内存

    try:
        with open(filepath, "rb") as f:
            # ✅ 修复：不使用海象运算符（Python 3.8 不支持）
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)

            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


# ✅ 修复：使用 Tuple[str] 而不是 tuple[str]
def listdir_with_allowed_type(path: str, allowed_types: Tuple[str]) -> Tuple[str]:
    """
    返回文件夹内允许的文件列表。

    Args:
        path: 文件夹路径
        allowed_types: 允许的文件后缀元组（如 (".pdf", ".txt")）

    Returns:
        文件路径的元组
    """
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return ()  # ✅ 修复：返回空元组，而不是 allowed_types

    files = []
    for f in os.listdir(path):
        # str.endswith() 可以直接接受元组参数
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


# ✅ 修复：使用 List[Document] 而不是 list[Document]
def pdf_loader(filepath: str, passwd=None) -> List[Document]:
    """加载 PDF 文件，返回 Document 列表。"""
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> List[Document]:
    """加载 TXT 文件，返回 Document 列表。"""
    return TextLoader(filepath, encoding="utf-8").load()