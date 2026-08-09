"""模型工厂：集中创建低输出预算的聊天与嵌入模型。"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatTongyi:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            temperature=float(rag_conf.get("temperature", 0.1)),
            max_tokens=int(rag_conf.get("max_tokens", 220)),
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> DashScopeEmbeddings:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

__all__ = ["BaseModelFactory", "ChatModelFactory", "EmbeddingsFactory", "chat_model", "embed_model"]
