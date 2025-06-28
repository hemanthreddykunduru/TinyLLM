import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import mimetypes
import fitz  # PyMuPDF
import docx
import pandas as pd
from urllib.parse import urlparse

SAVE_FILE = "web_scraped.txt"
QUERY = "artificial intelligence"
MAX_RESULTS = 100

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

def clean_text(text):
    return ' '.join(text.split())

def save_text(text):
    if text:
        with open(SAVE_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n" + "="*100 + "\n")

def extract_text_from_html(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return clean_text(text)
    except Exception as e:
        print(f"[HTML] Error at {url}: {e}")
        return None

def extract_text_from_pdf(url):
    try:
        res = requests.get(url, timeout=10)
        with open("temp.pdf", "wb") as f:
            f.write(res.content)
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove("temp.pdf")
        return clean_text(text)
    except Exception as e:
        print(f"[PDF] Error at {url}: {e}")
        return None

def extract_text_from_docx(url):
    try:
        res = requests.get(url, timeout=10)
        with open("temp.docx", "wb") as f:
            f.write(res.content)
        doc = docx.Document("temp.docx")
        text = "\n".join([p.text for p in doc.paragraphs])
        os.remove("temp.docx")
        return clean_text(text)
    except Exception as e:
        print(f"[DOCX] Error at {url}: {e}")
        return None

def extract_text_from_csv(url):
    try:
        df = pd.read_csv(url)
        return clean_text(df.to_string())
    except Exception as e:
        print(f"[CSV] Error at {url}: {e}")
        return None

def get_mime_type(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        return r.headers.get("Content-Type", "").split(";")[0]
    except:
        return ''

def extract_and_save(url):
    mime = get_mime_type(url)
    print(f"Processing: {url} [Type: {mime}]")

    ext = urlparse(url).path.lower()
    text = None

    if ".pdf" in ext or mime == "application/pdf":
        text = extract_text_from_pdf(url)
    elif ".docx" in ext or "wordprocessingml.document" in mime:
        text = extract_text_from_docx(url)
    elif ".csv" in ext or "text/csv" in mime:
        text = extract_text_from_csv(url)
    elif "html" in mime or url.startswith("http"):
        text = extract_text_from_html(url)

    save_text(text)

def main():
    with DDGS() as ddgs:
        print(f"Searching DuckDuckGo for: {QUERY}")
        results = ddgs.text(QUERY, max_results=MAX_RESULTS)
        urls = [r["href"] for r in results if r.get("href")]

        for url in urls:
            extract_and_save(url)

if __name__ == "__main__":
    main()
