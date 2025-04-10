import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import html2text
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Set
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

start_url = "https://www.fib.upc.edu/"
visited_urls: Set[str] = set()
output_folder = "downloaded_pages"
markdown_folder = "markdown_pages"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(markdown_folder, exist_ok=True)

def save_page(url: str, content: str) -> None:
    """Guarda el contenido HTML y Markdown de una página"""
    if not url.startswith(start_url):
        return

    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    if not path:
        path = "index"
    else:
        path = os.path.join(*path.split("/"))

    html_filepath = os.path.join(output_folder, f"{path}.html")
    md_filepath = os.path.join(markdown_folder, f"{path}.md")

    os.makedirs(os.path.dirname(html_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(md_filepath), exist_ok=True)

    try:
        soup = BeautifulSoup(content, "html.parser")
        
        # Actualizar enlaces relativos
        for link in soup.find_all("a", href=True):
            link_url = urljoin(url, link["href"])
            link_parsed_url = urlparse(link_url)
            link_path = link_parsed_url.path.strip("/")
            if not link_path:
                link_path = "index"
            else:
                link_path = os.path.join(*link_path.split("/"))
            link["href"] = f"{link_path}.md"

        updated_content = str(soup)

        # Guardar HTML
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(updated_content)
        logging.info(f"Saved HTML: {html_filepath}")

        # Convertir y guardar Markdown
        markdown_content = html2text.html2text(updated_content)
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info(f"Saved Markdown: {md_filepath}")

    except Exception as e:
        logging.error(f"Error saving page {url}: {str(e)}")

def fetch_with_retry(url: str, max_retries: int = 3) -> str:
    """Intenta descargar una URL con reintentos"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return ""

def crawl(url: str) -> None:
    """Crawlea una URL y sus enlaces"""
    if url in visited_urls or not url.startswith(start_url):
        return
    
    logging.info(f"Crawling: {url}")
    visited_urls.add(url)

    try:
        content = fetch_with_retry(url)
        if not content:
            return

        save_page(url, content)
        soup = BeautifulSoup(content, "html.parser")
        links_to_crawl = []

        # Extraer enlaces
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            if full_url not in visited_urls and full_url.startswith(start_url):
                links_to_crawl.append(full_url)

        # Procesar enlaces en paralelo
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
            executor.map(crawl, links_to_crawl)

    except Exception as e:
        logging.error(f"Error crawling {url}: {str(e)}")

if __name__ == "__main__":
    crawl(start_url)