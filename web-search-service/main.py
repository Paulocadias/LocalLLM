"""
Web Search Service - Real-time web search for your Local LLM
Uses DuckDuckGo (free, no API key needed)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from duckduckgo_search import DDGS
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Web Search Service", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    region: str = "wt-wt"  # Worldwide


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time: float


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "web-search"}


@app.post("/search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """
    Search the web using DuckDuckGo
    Returns top results with title, URL, and snippet
    """
    try:
        start_time = time.time()
        logger.info(f"Searching for: {request.query}")

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(
                request.query,
                region=request.region,
                safesearch="moderate",
                max_results=request.max_results
            )

            for result in search_results:
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("href", ""),
                    snippet=result.get("body", "")
                ))

        search_time = time.time() - start_time
        logger.info(f"Found {len(results)} results in {search_time:.2f}s")

        return SearchResponse(
            query=request.query,
            results=results,
            search_time=search_time
        )

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search_and_summarize")
async def search_and_summarize(request: SearchRequest, model_endpoint: str = "http://localllm:8080"):
    """
    Search web and have LLM summarize the results
    Returns answer with sources
    """
    try:
        # 1. Search the web
        search_response = await web_search(request)

        if not search_response.results:
            return {
                "answer": "No results found for your query.",
                "sources": [],
                "query": request.query
            }

        # 2. Build context from search results
        context_parts = []
        sources = []

        for i, result in enumerate(search_response.results, 1):
            context_parts.append(f"Source {i}: {result.title}\n{result.snippet}\nURL: {result.url}\n")
            sources.append({"title": result.title, "url": result.url})

        context = "\n".join(context_parts)

        # 3. Create prompt for LLM
        prompt = f"""Based on these web search results, answer the question comprehensively.

Question: {request.query}

Search Results:
{context}

Please provide a clear, accurate answer based on the search results above. Include relevant information and cite which sources you're using."""

        return {
            "prompt": prompt,
            "sources": sources,
            "query": request.query,
            "note": "Send this prompt to your LLM endpoint to get summarized answer"
        }

    except Exception as e:
        logger.error(f"Search and summarize error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
