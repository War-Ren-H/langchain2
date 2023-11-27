import os
import re
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

#query = "What if the debtor refuses to testify about the bankruptcy?"

def source_name(docs):
    """
    Create document.metadata['law'] to direct filename only
    use regex to isolate filename (w/o pdf)
    """
    #splits filename by /'s
    split_path = re.split(r'\/', docs.metadata['source'])
    #takes the last item from the split (the item name) and gets rid of the ".pdf" extension
    docs.metadata['law'] = re.split(r'\.',split_path[-1])[0]

documents = PyPDFDirectoryLoader("/Users/gv658da/Documents/tax_folder")
texts = documents.load_and_split()

for text in texts:
    source_name(text)

retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


#llm = OpenAI(temperature=0)
#chain = load_summarize_chain(llm, chain_type="map_reduce")
#chain.run(texts)

def find_relevant_docs(query):
    compressed_docs = compression_retriever.get_relevant_documents(query)
    return f"\n{'-' * 100}\n".join([d.metadata['law'] + 
                                   f"\n\n" + "page: " + str(d.metadata['page']) + 
                                   f"\n\n" + d.page_content for i, d in enumerate(compressed_docs)])

demo = gr.Interface(
    fn=find_relevant_docs,
    inputs=["text"],
    outputs=["text"],
)
demo.launch()