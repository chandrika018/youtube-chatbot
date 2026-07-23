import requests
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
import urllib.parse
from typing import Dict, List, Any
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def clean_text(text: str) -> str:
    # Remove excessive whitespaces and empty lines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def search_ddg(query: str, num_results: int = 4) -> List[Dict[str, str]]:
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    results = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"DuckDuckGo search failed with code {response.status_code}")
            return results
            
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", class_="result__url")
        snippets = soup.find_all("a", class_="result__snippet")
        titles = soup.find_all("a", class_="result__snippet")  # Fallback titles or classes
        
        # In DDG HTML interface, class result__snippet holds snippet, class result__title is a header
        title_elements = soup.find_all("h2", class_="result__title")
        
        for i in range(min(len(title_elements), num_results)):
            t_el = title_elements[i].find("a")
            if not t_el:
                continue
            
            raw_url = t_el.get("href", "")
            # Clean URL: DDG redirects urls through uddg parameter
            parsed_url = urllib.parse.urlparse(raw_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            url_target = raw_url
            if "uddg" in query_params:
                url_target = query_params["uddg"][0]
                
            title = t_el.get_text(strip=True)
            
            snippet_text = ""
            if i < len(snippets):
                snippet_text = snippets[i].get_text(strip=True)
                
            results.append({
                "title": title,
                "url": url_target,
                "snippet": snippet_text
            })
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        
    return results

def scrape_page_content(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing spaces
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text_content = "\n".join(chunk for chunk in chunks if chunk)
        
        # Return truncated clean version (up to 4000 characters)
        return clean_text(text_content)[:4000]
    except Exception as e:
        print(f"Failed to scrape page {url}: {e}")
        return ""

def search_and_synthesize(query: str) -> Dict[str, Any]:
    print(f"Performing fallback web search for: '{query}'")
    search_results = search_ddg(query)
    
    context_chunks = []
    sources = []
    
    for item in search_results:
        url = item["url"]
        title = item["title"]
        snippet = item["snippet"]
        
        # Skip certain domains to keep results clean (e.g., youtube itself, logins, etc.)
        if "youtube.com" in url or "youtu.be" in url or "accounts.google" in url:
            continue
            
        scraped_text = scrape_page_content(url)
        if not scraped_text or len(scraped_text) < 150:
            # Fall back to snippet if scraping fails or is too thin
            scraped_text = snippet
            
        context_chunks.append(f"Source: {title} ({url})\nContent: {scraped_text}")
        sources.append({
            "title": title,
            "url": url
        })
        
        # Scrape maximum 3 successful pages to keep performance high and cost low
        if len(context_chunks) >= 3:
            break
            
    # Combine content chunks
    combined_context = "\n\n---\n\n".join(context_chunks)
    
    return {
        "context": combined_context if combined_context else "No content found from web search.",
        "sources": sources,
        "query": query
    }
