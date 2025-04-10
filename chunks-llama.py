import os
import markdown
import requests
import json
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Tuple, Dict, Any
from datetime import datetime
import time

# ==============================
# 1️⃣ Extraer texto de archivos Markdown
# ==============================
def extract_text_from_markdown(md_file: str) -> str:
    """Leer archivo Markdown y convertirlo en texto plano preprocesado"""
    try:
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extraer texto y preprocesarlo
        text = soup.get_text()
        return preprocess_text(text)
    except Exception as e:
        print(f"Error procesando archivo {md_file}: {str(e)}")
        return ""

def load_markdown_files(folder):
    """ Leer recursivamente todos los archivos Markdown """
    all_texts = []
    file_sources = []  # Lista para guardar las fuentes de origen
    
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, folder)
                text = extract_text_from_markdown(file_path)
                all_texts.append(text)
                file_sources.append(f"Archivo: {relative_path}")
    
    return all_texts, file_sources

# ==============================
# 2️⃣ Dividir textos y calcular embeddings
# ==============================
def chunk_texts(texts: List[str], chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
    """Divide el texto en fragmentos optimizados para mantener el contexto"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = []
    for text in texts:
        if text.strip():  # Solo procesar textos no vacíos
            text_chunks = text_splitter.split_text(text)
            # Filtrar chunks demasiado pequeños
            chunks.extend([chunk for chunk in text_chunks if len(chunk) > 100])
    
    print(f"✅ Generados {len(chunks)} chunks con tamaño {chunk_size} y solapamiento {chunk_overlap}")
    return chunks

def compute_embeddings(chunks):
    """ Calcular embeddings """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, model

# ==============================
# 3️⃣ Almacenar en ChromaDB
# ==============================
def store_in_chromadb(chunks: List[str], embeddings: List[Any], sources: List[str] = None) -> None:
    """Almacena texto y embeddings en ChromaDB con metadatos mejorados"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(
            name="markdown_docs",
            metadata={"hnsw:space": "cosine"}  # Usar similitud coseno para mejor precisión
        )
        
        # Verificar si ya existen datos
        existing_count = collection.count()
        if existing_count > 0:
            print(f"⚠️ Ya existen {existing_count} documentos en la base de datos")
            user_input = input("¿Desea actualizar la base de datos? (s/n): ")
            if user_input.lower() != 's':
                return
        
        # Generar IDs únicos y metadatos mejorados
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            source = sources[i] if sources and i < len(sources) else f"Fragmento {i}"
            metadata = normalize_metadata(source)
            metadata.update({
                "chunk_id": str(i),
                "chunk_length": str(len(chunk)),
                "timestamp": str(datetime.now().isoformat())
            })
            metadatas.append(metadata)
        
        # Añadir chunks a ChromaDB en lotes para mejor rendimiento
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = [emb.tolist() for emb in embeddings[i:i + batch_size]]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = chunks[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        
        print(f"✅ {len(chunks)} fragmentos almacenados en ChromaDB")
        print(f"📊 Estadísticas: {collection.count()} documentos totales")
        
    except Exception as e:
        print(f"❌ Error al almacenar en ChromaDB: {str(e)}")
        raise

# ==============================
# 4️⃣ Consultar ChromaDB
# ==============================
def query_chromadb(query_text: str, model: Any, top_n: int = 15) -> str:
    """Consulta ChromaDB para encontrar los fragmentos más relevantes con mejor filtrado"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="markdown_docs")
        
        # Preprocesar la consulta manteniendo mayúsculas
        processed_query = preprocess_text(query_text)
        query_embedding = model.encode([processed_query])[0].tolist()
        
        # Realizar la consulta con más resultados
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n * 3,  # Obtener más resultados para mejor filtrado
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or len(results["documents"][0]) == 0:
            return "No se encontró información relevante para tu consulta."
        
        # Construir contexto con mejor filtrado y ranking
        context_parts = []
        seen_sources = set()
        
        for i, (doc, metadata, distance) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
            # Umbral de distancia más permisivo
            if distance < 2.5 and len(context_parts) < top_n:
                source = metadata.get("source", "Desconocido")
                
                # Permitir múltiples chunks de la misma fuente si son relevantes
                relevance_score = 1 - (distance / 2.5)  # Normalizar score entre 0 y 1
                if relevance_score > 0.3:  # Umbral mínimo de relevancia
                    context_parts.append({
                        "text": doc,
                        "source": source,
                        "relevance": relevance_score
                    })
        
        # Ordenar por relevancia
        context_parts.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Construir respuesta formateada
        formatted_context = []
        for part in context_parts:
            formatted_context.append(
                f"[Fuente: {part['source']} - Relevancia: {part['relevance']:.2f}]\n{part['text']}"
            )
        
        return "\n\n".join(formatted_context) if formatted_context else "No se encontró información suficientemente relevante."
        
    except Exception as e:
        print(f"❌ Error al consultar ChromaDB: {str(e)}")
        return "Hubo un error al procesar tu consulta."

# ==============================
# 5️⃣ Inicializar el modelo LMStudio y generar respuesta
# ==============================
# Configurar la URL de la API de LMStudio
LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

def ask_ai_with_context(question: str, model: Any) -> str:
    """Consulta ChromaDB y genera una respuesta con el modelo local LMStudio"""
    try:
        context = query_chromadb(question, model)
        
        # Prompt mejorado y estructurado
        prompt = f"""Eres un asistente experto en la UPC (Universitat Politècnica de Catalunya).
Tu tarea es responder preguntas utilizando ÚNICAMENTE la información proporcionada en el contexto.
Si la información no está en el contexto, indica claramente que no tienes esa información.
No inventes ni añadas información que no esté en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
1. Analiza cuidadosamente el contexto proporcionado.
2. Identifica los fragmentos más relevantes para la pregunta.
3. Proporciona una respuesta clara y concisa basada ÚNICAMENTE en el contexto dado.
4. Cita las fuentes específicas del contexto que utilizaste para tu respuesta.
5. Si la información en el contexto es insuficiente, indícalo claramente.
6. Si encuentras información contradictoria en el contexto, menciona las diferentes fuentes.
7. Mantén un tono profesional y objetivo.
8. Si la pregunta es ambigua, pide clarificación.

RESPUESTA:"""
        
        # Configuración mejorada para la API de LMStudio
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Eres un asistente experto en la UPC que responde preguntas basándose únicamente en el contexto proporcionado."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3,  # Más determinista para respuestas más precisas
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            LMSTUDIO_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30  # Añadir timeout para evitar bloqueos
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            error_msg = f"Error en la API de LMStudio: {response.status_code} - {response.text}"
            print(error_msg)
            return f"Lo siento, hubo un error al procesar tu pregunta: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Lo siento, la solicitud tardó demasiado en procesarse. Por favor, intenta de nuevo."
    except Exception as e:
        print(f"Error al llamar a la API local de LMStudio: {str(e)}")
        return "Lo siento, hubo un error al procesar tu pregunta con el modelo local."

# ==============================
# 6️⃣ Ejecutar el flujo
# ==============================
def main():
    def log_step(step_name: str, start_time: float = None) -> float:
        """Registra el tiempo de ejecución de cada paso"""
        current_time = time.time()
        if start_time:
            duration = current_time - start_time
            print(f"⏱️  {step_name} completado en {duration:.2f} segundos")
        else:
            print(f"\n🔄 Iniciando {step_name}...")
        return current_time
    
    # Ruta a la carpeta con archivos Markdown
    markdown_folder = "./markdown_pages"
    
    # Procesamiento inicial
    print("\n🚀 Iniciando procesamiento de documentos...")
    start_time = time.time()
    
    # 1. Cargar archivos Markdown
    step_start = log_step("Carga de archivos Markdown")
    texts, sources = load_markdown_files(markdown_folder)
    log_step("Carga de archivos Markdown", step_start)
    print(f"📚 Se cargaron {len(texts)} archivos Markdown")
    
    # 2. Dividir textos en chunks
    step_start = log_step("División en chunks")
    chunks = chunk_texts(texts)
    log_step("División en chunks", step_start)
    print(f"✂️  Se generaron {len(chunks)} chunks")
    
    # 3. Calcular embeddings
    step_start = log_step("Cálculo de embeddings")
    embeddings, model = compute_embeddings(chunks)
    log_step("Cálculo de embeddings", step_start)
    print(f"🔢 Se calcularon embeddings para {len(embeddings)} chunks")
    
    # 4. Almacenar en ChromaDB
    step_start = log_step("Almacenamiento en ChromaDB")
    store_in_chromadb(chunks, embeddings, sources)
    log_step("Almacenamiento en ChromaDB", step_start)
    
    # Tiempo total de procesamiento inicial
    total_time = time.time() - start_time
    print(f"\n✅ Procesamiento inicial completado en {total_time:.2f} segundos")
    
    # Bucle de preguntas interactivo
    print("\n✨ Asistente DeepSeek UPC listo para responder preguntas ✨")
    print("📝 Escribe 'salir' para terminar.")
    
    while True:
        question = input("\n❓ Pregunta: ")
        if question.lower() == "salir":
            break
            
        # Medir tiempo de respuesta
        start_time = time.time()
        print("\n🔄 Procesando pregunta...")
        
        # 1. Consulta a ChromaDB
        step_start = log_step("Búsqueda en ChromaDB")
        context = query_chromadb(question, model)
        log_step("Búsqueda en ChromaDB", step_start)
        
        # 2. Generación de respuesta
        step_start = log_step("Generación de respuesta")
        answer = ask_ai_with_context(question, model)
        log_step("Generación de respuesta", step_start)
        
        # Tiempo total de respuesta
        total_time = time.time() - start_time
        print(f"\n🤖 Respuesta (generada en {total_time:.2f} segundos):")
        print(answer)

def preprocess_text(text: str) -> str:
    """Preprocesa el texto para mejorar la calidad de los embeddings"""
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text)
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Mantener caracteres especiales importantes
    text = re.sub(r'[^\w\s.,?¿¡!()\-]', '', text)
    return text.strip()

def normalize_metadata(source: str) -> Dict[str, str]:
    """Normaliza los metadatos para mantener consistencia"""
    return {
        "source": source,
        "type": "markdown",
        "processed": "true"
    }

if __name__ == "__main__":
    main()
