

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text = "This is the sample text to be embedded."
emb = embeddings.embed_query(text)
print(f"Embedded Text: {emb}")
print(f"Embedding Length: {len(emb)}") # 1536

# mutiple texts

texts = ["This is the first text to be embeeded.", 
         "This is the second text to be embeeded."]

embs = embeddings.embed_documents(texts)
print(f"Number of embeddings: {len(embs)}")


# free embeddings
# embeddings = HuggingFaceEmbeddings(
#       model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions

# Ollama 
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0")


#OpenAI embeddings
# Model	                    Dimensions	    Cost per 1M tokens	    Best For
# text-embedding-3-small	    1536	        $0.02	            General use
# text-embedding-3-large	    3072	        $0.13	            High accuracy
# text-embedding-ada-002	    1536	        $0.10	            Legacy










