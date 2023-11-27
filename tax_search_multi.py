import os
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

query = "What if the debtor refuses to testify about the bankruptcy?"

def source_name(docs):
    """
    Create document.metadata['law'] to direct filename only
    use regex to isolate filename (w/o pdf)
    """
    #splits filename by /'s
    split_path = re.split(r'\/', docs.metadata['source'])
    #takes the last item from the split (the item name) and gets rid of the ".pdf" extension
    docs.metadata['law'] = re.split(r'\.',split_path[-1])[0]

def pretty_print_docs(docs):
    """
    Add something to print in the metadata before the page content as well
    """
    print(f"\n{'-' * 100}\n".join([d.metadata['law'] + 
                                   f"\n\n" + "page: " + str(d.metadata['page']) + 
                                   f"\n\n" + d.page_content for i, d in enumerate(docs)]))


documents = PyPDFDirectoryLoader("/Users/gv658da/Documents/tax_folder")
texts = documents.load_and_split()

for text in texts:
    source_name(text)
    #print(text.metadata)

retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents(query)
pretty_print_docs(compressed_docs)

testing = FAISS.from_documents(texts, OpenAIEmbeddings()).similarity_search_with_score(query)
#print(testing[0][-1])
#print(len(testing))
for i in range(len(testing)):
    print(testing[i][-2])
    print(testing[i][-1])
    print('\n')
    # print(testing[i][-1]['metadata'])
#print(compression_retriever)


findit = re.search(r'page_content=\'(.*?)\' metadata', str(testing[0][0])).group(1)

print(findit)

findit2 = re.search(r'page_content=\'(.*?)\' metadata', str(testing[0][0])).group(1)


print(type(compressed_docs))

from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(compressed_docs)