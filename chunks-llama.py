import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_knowledge")

# 存入信息
collection.add(ids=["1"], documents=["这是你的信息"])

# 查询相关信息
results = collection.query(query_texts=["你的问题"], n_results=1)
print(results["documents"])