from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader

class DocumentationScraper:
    def __init__(self, homepage_url):
        self.homepage_url = homepage_url
        self.doc_links = []
        self.docs = []
        self._setup_selenium_options()

    def _setup_selenium_options(self):
        """Set up Selenium WebDriver options"""
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")

    def get_documentation_links(self):
        """Extract all documentation links from the homepage"""
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options
        )
        
        driver.get(self.homepage_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = set()

        for a_tag in soup.find_all("a", href=True):
            link = urljoin(self.homepage_url, a_tag["href"])
            parsed_link = urlparse(link)
            
            if (parsed_link.netloc == urlparse(self.homepage_url).netloc and 
                'enterprise' not in link.lower()):
                links.add(link)

        driver.quit()
        self.doc_links = list(links)
        return self.doc_links

    def load_documents(self):
        """Load documents using WebBaseLoader"""
        if not self.doc_links:
            self.get_documentation_links()
        
        loader = WebBaseLoader(self.doc_links)
        self.docs = loader.load()
        return self.docs

    def save_content(self, filename="doc_sel.txt"):
        """Save documents content to a file"""
        if not self.docs:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        content = ""
        for doc in self.docs:
            content += doc.page_content
            
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

def main():
    scraper = DocumentationScraper("https://dash.plotly.com/")
    links = scraper.get_documentation_links()
    print(f"Found {len(links)} documentation links.")
    
    docs = scraper.load_documents(num_links=4)
    print(f"Loaded {len(docs)} documents.")
    
    # scraper.save_content_to_file()

if __name__ == "__main__":
    main()