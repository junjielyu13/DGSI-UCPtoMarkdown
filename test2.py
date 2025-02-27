import os
import requests
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin, urlparse

# Dominio base y URL inicial
BASE_DOMAIN = "upc.edu"
START_URL = "https://www.upc.edu/"

# Conjunto para llevar el control de URLs visitadas y evitar ciclos
visited = set()

def save_markdown(url, markdown_text):
    """
    Guarda el contenido Markdown en una estructura de carpetas basada en la URL.
    """
    parsed = urlparse(url)
    # Se utiliza el path; si está vacío se usa "index"
    path = parsed.path.strip("/") or "index"
    
    # Si el path termina en /, se asume index para esa carpeta
    if path.endswith("/"):
        path = os.path.join(path, "index")
    
    # Se agrega la extensión .md y se coloca en la carpeta 'output'
    file_path = os.path.join("output", path + ".md")
    
    # Crear las carpetas necesarias
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Guardado: {file_path}")

def crawl(url):
    """
    Recorre la URL, convierte el HTML a Markdown, guarda el resultado y recurre en enlaces internos.
    """
    if url in visited:
        return
    visited.add(url)
    
    try:
        response = requests.get(url)
    except Exception as e:
        print(f"Error al acceder a {url}: {e}")
        return
    
    if response.status_code != 200:
        print(f"No se pudo acceder a {url} (código {response.status_code})")
        return

    html = response.text
    # Convertir HTML a Markdown
    markdown = html2text.html2text(html)
    save_markdown(url, markdown)

    # Parsear el HTML para encontrar enlaces
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        link = a["href"]
        # Crear la URL absoluta
        full_link = urljoin(url, link)
        parsed_link = urlparse(full_link)
        
        # Verificar que el enlace pertenezca al dominio deseado y no sea un ancla
        if parsed_link.netloc.endswith(BASE_DOMAIN) and not full_link.endswith("#"):
            crawl(full_link)

if __name__ == "__main__":
    crawl(START_URL)
