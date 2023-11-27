from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

documents = PyPDFDirectoryLoader("/Users/gv658da/Documents/tax_folder")
texts = documents.load_and_split()

print(type(texts))

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 500,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)

nu_docs = []

for text in texts:
    words = dict(text)['page_content']
    print(words)
    #break
    nu_docs.append(text_splitter.create_documents([words]))

print(nu_docs[0])
# texts = text_splitter.create_documents([texts])
#print(texts[0])
#print(texts[1])