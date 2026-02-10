import os
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load ASU Library documents
documents = SimpleDirectoryReader("asu_data").load_data()
texts = [doc.text for doc in documents]

# 2. Create embedding model
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# 3. Create Chroma client
client = chromadb.PersistentClient(path="./llamachromadb")

# 4. Delete ASU collection if it exists (clean slate)
try:
    client.delete_collection(name="asulib")
except:
    pass

# 5. Create ASU collection
collection = client.create_collection(name="asulib")

# 6. Embed text and add to Chroma EXPLICITLY
embeddings = embed_model.get_text_embedding_batch(texts)

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[f"asu_{i}" for i in range(len(texts))]
)

print(f"✅ Added {len(texts)} ASU documents to Chroma collection 'asulib'")
