from typing import ClassVar, List, Set
from qdrant_client.http.models import Filter
from langchain.tools import BaseTool
from V3.env import qdrant_client, collections, TOP_K
from V3.classes import GraphState, GraphStateManager


class VectorDatabaseRetrieveExtensionCodes(BaseTool):
    name: ClassVar[str] = "vector_database_retrieve_extension_codes"
    description: ClassVar[str] = (
        "Retrieves extension codes from the vector database based on the given stem codes."
    )