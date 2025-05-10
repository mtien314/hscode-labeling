from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import DeepInfra
from tqdm import tqdm
from huggingface_hub import login
import bm25s
from bm25s.hf import BM25HF
import os

token = ""
login(token)
os.environ["DEEPINFRA_API_TOKEN"] = ""

#setting model
llm = DeepInfra(model_id="google/gemma-3-27b-it")

# Load a BM25 index from the Hugging Face model hub
user = "tien314/hscode8-version5"
retriever = BM25HF.load_from_hub(f"{user}", load_corpus = True, mmap = True)

def process_retrieval(description):
  description = description.lower()
  # Tokenize the query
  query_tokens = bm25s.tokenize(description)
  # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
  results, scores = retriever.retrieve(query_tokens, k=10)
  #convert results to document
  documents = [Document(doc['text']) for doc in results[0]]
  return documents

def process_generate(description, documents):
  prompt = ChatPromptTemplate.from_messages([
          HumanMessagePromptTemplate.from_template(
          """
          Extract the appropriate 8-digit HS Code from the product description by thoroughly analyzing its details and utilizing a reliable and up-to-date HS Code database for accurate results.
          Only return the HS Code as a 8-digit number .
          Example: 12345678
          Context: {context}
          Description: {description}
          Answer:
          """
          )
      ])
      
  chain = prompt|llm

  hscode_generated = chain.invoke({'context': documents, 'description': description})
  return hscode_generated

if __name__ == "__main__":
  description = "chocolates"
  documents = process_retrieval(description)
  hscode_generated = process_generate(description,documents)
  print(hscode_generated)