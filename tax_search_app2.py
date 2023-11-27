from flask import Flask, render_template, request
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
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv


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

#findit = re.search(r'page_content=\'(.*?)\' metadata', str(testing[0][0])).group(1)

def pretty_print_docs(docs):
    """
    Add something to print in the metadata before the page content as well
    """
    return f"\n{'-' * 100}\n".join([f"\n\n" + "Similarity Score: " + str(d[0][-1]) + 
                                   f"\n\n" + re.search(r'page_content=\'(.*?)\' metadata', str(d[0][0])).group(1) for i, d in enumerate(docs)])




documents = PyPDFDirectoryLoader("/Users/gv658da/Documents/tax_folder")
texts = documents.load_and_split()

for text in texts:
    source_name(text)
    #print(text.metadata)

llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")
#chain.run(docs)

app = Flask(__name__)

@app.route('/')
def home():
    answer = ''
    #return render_template('index.html', answer=answer)
    return render_template('index.html', **locals())

@app.route('/answer', methods=['POST', 'GET'])
def predict():
    query = request.form['query']
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).similarity_search_with_score(query)
    answer = pretty_print_docs(retriever)
    #print(type(answer))
    #return render_template('index.html', answer=answer)
    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)