import os
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def extract_text_from_markdown(md_file):
    """ 读取 Markdown 文件并提取纯文本 """
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # 转换为 HTML 并去除格式
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def load_markdown_files(folder):
    """ 递归读取文件夹及其子文件夹中的所有 Markdown 文件 """
    all_texts = []
    
    for root, _, files in os.walk(folder):  # 使用 os.walk 递归遍历所有子文件夹
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                text = extract_text_from_markdown(file_path)
                all_texts.append(text)
                
    return all_texts

def chunk_texts(texts, chunk_size=500, chunk_overlap=50):
    """ 拆分文本为 Chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def compute_embeddings(chunks):
    """ 使用本地模型计算嵌入 """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, model

def store_in_chromadb(chunks, embeddings):
    """ 存入 ChromaDB """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="markdown_docs")
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": f"Markdown Chunk {i}"}],
            documents=[chunk]
        )
    print("数据存入 ChromaDB 成功！")

def query_chromadb(query_text, model, top_n=3):
    """ 在 ChromaDB 进行相似性搜索 """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="markdown_docs")
    query_embedding = model.encode([query_text])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)
    
    for doc, score in zip(results["documents"][0], results["distances"][0]):
        print(f"相关 Chunk: {doc} (相似度: {score})")

if __name__ == "__main__":
    folder_path = "./markdown_pages"  # 你的 Markdown 文件夹路径
    texts = load_markdown_files(folder_path)
    chunks = chunk_texts(texts)
    embeddings, model = compute_embeddings(chunks)
    store_in_chromadb(chunks, embeddings)
    
    # 测试搜索
    query_text = "什么是 ChromaDB？"
    print("\n🔍 搜索结果:")
    query_chromadb(query_text, model)