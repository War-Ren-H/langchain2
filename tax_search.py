from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFLoader
import os

os.environ["OPEN_API_KEY"] = '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'


##Change the file later
loader = PyPDFLoader("/Users/gv658da/Documents/taxlaw1.pdf")
pages = loader.load_and_split()

index = VectorstoreIndexCreator().from_loaders([loader])

query = "What if the debtor refuses to testify about the bankruptcy?"
print(index.query_with_sources(query))

class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """

