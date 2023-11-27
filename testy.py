from PIL import Image

#image_1 = Image.open(r'path where the image is stored\file name.png')
#im_1 = image_1.convert('RGB')
#im_1.save(r'path where the pdf will be stored\new file name.pdf')


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

documents = PyPDFDirectoryLoader("/Users/gv658da/Documents/tax_folder")
texts = documents.load_and_split()
#llm = OpenAI(temperature=0)

#from langchain.chains.summarize import load_summarize_chain
#chain = load_summarize_chain(llm, chain_type="map_reduce")
#print(chain.run(texts))

import PyPDF2
 
# creating a pdf file object
pdfFileObj = open('/Users/gv658da/Documents/taxlaw1.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)

# printing number of pages in pdf file
print(len(pdfReader.pages))
 
# creating a page object
page_objs = []
for i in range(len(pdfReader.pages)):
    page_objs.append(pdfReader.pages[i])

for p in page_objs:
    print(p)
    print('--------------------------')
    print('\n')