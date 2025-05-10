from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import DeepInfra
import asyncio 
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
  # Tokenize the query
  query_tokens = bm25s.tokenize(description)
  # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
  results, scores = retriever.retrieve(query_tokens, k=10)
  #convert results to document
  documents = [[Document(d['text']) for d in doc] for doc in results]
  return documents

def initialize_chain():
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
  return chain

async def generate_hscode_single(chain, context, description):
    return await chain.ainvoke({"context": context, "description": description})

async def generate_hscode_in_batches_async(chain, documents_list, descriptions_list, batch_size=5):
    results = []

    for i in range(0, len(documents_list), batch_size):
        batch_docs = documents_list[i:i + batch_size]
        batch_descs = descriptions_list[i:i + batch_size]

        tasks = [
            generate_hscode_single(chain, doc, desc)
            for doc, desc in zip(batch_docs, batch_descs)
        ]

        # Run tasks concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                print("Error:", result)
                results.append(None)
            else:
                results.append(result)

    return results

async def run_generate(chain, documents_list, descriptions_list):
    return await generate_hscode_in_batches_async(chain, documents_list, descriptions_list)


async def async_main():
   #example
    descriptions_list = ["chocolates", "live horses"] * 10
    documents_list = process_retrieval(descriptions_list)  
    chain = initialize_chain()  
    results = await run_generate(chain, documents_list, descriptions_list)
    return results

if __name__ == "__main__":
   results = asyncio.run(async_main())
   print(results)
