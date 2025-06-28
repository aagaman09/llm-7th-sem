from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
import os
from dotenv import load_dotenv

class GoogleSearchLinkAgent:
    """
    A class to perform Google searches and extract the top N links.
    """
    def __init__(self, num_results=3):
        """
        Initializes the GoogleSearchLinkExtractor.

        Args:
            num_results (int): The number of top search results to retrieve links from.
                               Defaults to 3.
        """
        load_dotenv()
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.api_key = os.getenv('GOOGLE_API_KEY')

        if not self.google_cse_id or not self.api_key:
            raise ValueError("Google CSE ID and API Key must be set in environment variables.")

        os.environ["GOOGLE_CSE_ID"] = self.google_cse_id
        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.search = GoogleSearchAPIWrapper()
        self.num_results = num_results
        self.tool = self._create_tool()

    def _top_n_results(self, query: str) -> list:
        """
        Internal method to get the top N search results.

        Args:
            query (str): The search query.

        Returns:
            list: A list of search result dictionaries.
        """
        return self.search.results(query, self.num_results)

    def _create_tool(self) -> Tool:
        """
        Creates and returns a Langchain Tool for Google Search.

        Returns:
            Tool: A Langchain Tool configured for Google Search.
        """
        return Tool(
            name="Google Search Snippets",
            description=f"Search Google for recent results and return top {self.num_results} snippets.",
            func=self._top_n_results,
        )

    def get_links(self, query: str) -> list[str]:
        """
        Performs a Google search and returns a list of links from the top results.

        Args:
            query (str): The search query provided by the user.

        Returns:
            list[str]: A list of URLs from the search results.
        """
        results = self.tool.run(query)
        links = [item['link'] for item in results if 'link' in item]
        return links

# Example Usage:
if __name__ == "__main__":
    try:
        # Initialize the extractor with default or custom number of results
        extractor = GoogleSearchLinkAgent(num_results=3) 
        # [Image of a magnifying glass searching the internet]

        user_query = input("Enter your search query: ")
        
        if user_query:
            extracted_links = extractor.get_links(user_query)
            print(f"\nTop {extractor.num_results} links for '{user_query}':")
            for link in extracted_links:
                print(link)
        else:
            print("No query entered. Exiting.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")