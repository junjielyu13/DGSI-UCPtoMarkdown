import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Función para realizar la consulta al LLM

def query_llm(prompt):
    url = "http://127.0.0.1:1234/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 1000,  # Incrementar el número de tokens para obtener respuestas más largas
        "stop": None,  # Eliminar el stop para permitir respuestas más largas
        "temperature": 0.7  # Ajustar la temperatura para respuestas más creativas
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    try:
        return response.json()["choices"][0]["text"].strip()
    except KeyError:
        return "Error: No se encontraron respuestas en la respuesta del LLM"

# Función para generar el prompt

def generate_prompt(question, context):
    return f"Responde la siguiente pregunta en catalán sobre la universidad UPC de Barcelona: {question}\nContexto: {context}"

# Función para realizar la consulta a ChromaDB

def query_chromadb(query_text, model, top_n=5):  # Incrementar el número de resultados para obtener más contexto
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="markdown_docs")
    query_embedding = model.encode([query_text])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)
    context = "\n".join(results["documents"][0])
    return context

# Función principal

def main():
    question = input("Introduce tu pregunta: ")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    context = query_chromadb(question, model)
    prompt = generate_prompt(question, context)
    answer = query_llm(prompt)
    print(f"Respuesta: {answer}")

if __name__ == "__main__":
    main()