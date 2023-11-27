import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

documents = PyPDFLoader("/Users/gv658da/Documents/taxlaw1.pdf")

texts = documents.load_and_split()
#documents = TextLoader('../../../state_of_the_union.txt').load()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#texts = text_splitter.split_documents(documents)

retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

docs = retriever.get_relevant_documents("What if the debtor refuses to testify about the bankruptcy?")
#pretty_print_docs(docs)

#from langchain.llms import OpenAI
#from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import LLMChainExtractor

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What if the debtor refuses to testify about the bankruptcy?")
pretty_print_docs(compressed_docs)

compressed_docs[0].metadata['test'] = 'test'

print(compressed_docs[0].metadata)