import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import html2text

# 目标网站
start_url = "https://www.fib.upc.edu/es"

# 记录已访问的URL
visited_urls = set()

# 存储 HTML 和 Markdown 的文件夹
output_folder = "downloaded_pages"
markdown_folder = "markdown_pages"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(markdown_folder, exist_ok=True)

def save_page(url, content):

    """Guardar la página HTML y convertirla a Markdown"""

    if not url.startswith(start_url):
        return

    parsed_url = urlparse(url)
    filename = parsed_url.path.strip("/").replace("/", "_") or "index"
    html_filepath = os.path.join(output_folder, f"{filename}.html")
    md_filepath = os.path.join(markdown_folder, f"{filename}.md")

    # Actualizar los enlaces en el contenido HTML
    soup = BeautifulSoup(content, "html.parser")
    for link in soup.find_all("a", href=True):
        link_url = urljoin(url, link["href"])
        link_parsed_url = urlparse(link_url)
        link_filename = link_parsed_url.path.strip("/").replace("/", "_") or "index"
        link["href"] = f"{link_filename}.md"

    updated_content = str(soup)

    # Guardar HTML actualizado
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print(f"Saved HTML: {html_filepath}")

    # Convertir a Markdown y guardar
    markdown_content = html2text.html2text(updated_content)
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved Markdown: {md_filepath}")

def crawl(url):
    """递归爬取网站并转换 HTML 为 Markdown"""
    if url in visited_urls or not url.startswith(start_url):
        return
    print(f"Crawling: {url}")
    visited_urls.add(url)
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 检查请求是否成功
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return
    
    save_page(url, response.text)
    
    soup = BeautifulSoup(response.text, "html.parser")

    # 提取所有链接
    for link in soup.find_all("a", href=True):
        full_url = urljoin(url, link["href"])
        if full_url not in visited_urls:
            crawl(full_url)  # 递归爬取

# 启动爬虫
crawl(start_url)