import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bm25s
from bm25s.hf import BM25HF
import os 


def process_data(df):
    """clean data include drop NULL,remove special character, lower case.... """
    df = df.dropna()
    descriptions = [des.lower() for  des in df['description']]
    df['description'] = descriptions
    return df

def process_chunking(df):
    documents = []

    for index, row in df.iterrows():
        text = row['description']
        hs_code = row['hs_code']
        documents.append(Document(page_content=text, metadata={'hs_code': hs_code}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  #size can be change
        chunk_overlap=0
    )

    split_documents = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    docs = []
    for doc in split_documents:
        metadata = doc.metadata
        metadata_str = str(metadata).strip('{}')
        page = doc.page_content
        docs.append([metadata_str + " " + page])

    cleaned_list = [item.replace('"', '').replace("'",'')  for items in docs for item in items]

    return cleaned_list

def create_BM25Model_local(cleaned_list,model_path):
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=cleaned_list)
    retriever.index(bm25s.tokenize(cleaned_list))
    ### Save the BM25 index to a file
    retriever.save(model_path, corpus=cleaned_list)

def load_BM25Model_local(model_path):
    ### Load the BM25 index as a memory-mapped file, which is memory efficient
    # and reduce overhead of loading the full index into memory
    retriever = bm25s.BM25.load(model_path, load_corpus = True,mmap=True)
    return retriever

def create_BM25_to_save_huggingface(cleaned_list, user):
    #create a BM25 Index
    retriever = BM25HF(corpus = cleaned_list)
    #create your corpus here
    corpus_token = bm25s.tokenize(cleaned_list)
    retriever.index(corpus_token)
    #set your name and token
    
    token = os.enivron['HF_TOKEN']
    retriever.save_to_hub(f"{user}", token = token, corpus=cleaned_list)

def load_BM25Model_from_huggingface(user):
    retriever = BM25HF.load_from_hub(f"{user}", load_corpus=True)
    return retriever


if __name__ == "__main__":
    path_data = ""
    #read data 
    df = pd.read_csv(path_data,dtype= {"hs_code":"object"})
    df = process_data(df)
    cleaned_list = process_chunking(df)
    ## Method 1: save bm25 local"
    model_path = ""
    create_BM25Model_local(cleaned_list,model_path)
    #load model
    retriever = load_BM25Model_local(model_path)
    ### method 2: save bm25 to huggingface
    user = ""
    create_BM25_to_save_huggingface(cleaned_list,user)
    retriever = load_BM25Model_from_huggingface(user)

    ## test 
    description = ""
    docs, scores = retriever.retrieve(bm25s.tokenize(description), k = 10)
    print(docs)
    