import asyncio
from crawl4ai import AsyncWebCrawler

class ScrapingAgent:
    """
    A scraping agent class to crawl a list of URLs and extract their Markdown content.
    """
    def __init__(self):
        """
        Initializes the ScrapingAgent.
        """
        print("ScrapingAgent initialized.")

    async def scrape_urls(self, urls: list[str]) -> dict[str, str]:
        """
        Asynchronously scrapes a list of URLs and returns their Markdown content.

        Args:
            urls: A list of strings, where each string is a URL to be scraped.

        Returns:
            A dictionary where keys are URLs and values are their Markdown content.
            Returns an empty string if scraping fails for a specific URL.
        """
        scraped_data = {}
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                print(f"\n--- Scraping: {url} ---")
                try:
                    result = await crawler.arun(url=url)
                    scraped_data[url] = result.markdown
                    print(f"--- Successfully scraped: {url} ---")
                except Exception as e:
                    print(f"--- Failed to scrape {url}: {e} ---")
                    scraped_data[url] = "" # Store empty string or error message if scraping fails
            print("\nAll scraping tasks completed.")
        return scraped_data

# --- How to use the ScrapingAgent class ---
async def main():
    # Define the list of URLs to scrape
    urls_to_scrape = [
        "https://ai.pydantic.dev/",
        "https://python.langchain.com/docs/integrations/chat/ollama/",
        "https://github.com/Anjila-26/FullStack_RAG"
    ]

    # Create an instance of the ScrapingAgent
    agent = ScrapingAgent()

    # Call the asynchronous scraping method
    results = await agent.scrape_urls(urls_to_scrape)

    # Process and print the results
    print("\n--- Scraping Results Summary ---")
    for url, markdown_content in results.items():
        if markdown_content:
            print(f"\n[SUCCESS] URL: {url}\nContent Snippet:\n{markdown_content[:500]}...\n") # Print first 500 chars
        else:
            print(f"\n[FAILED] URL: {url} - No content retrieved.\n")

if __name__ == "__main__":
    asyncio.run(main())