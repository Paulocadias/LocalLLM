from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
reflection_requests_total = Counter('reflection_requests_total', 'Total reflection requests')
reflection_errors_total = Counter('reflection_errors_total', 'Total reflection errors')
reflection_duration = Histogram('reflection_duration_seconds', 'Reflection request duration')

app = FastAPI(title="Reflection Agent", version="1.0.0")

class ReflectionRequest(BaseModel):
    query: str
    context: str
    model_response: str

class ReflectionResponse(BaseModel):
    reflection: str
    improvements: list[str]
    confidence_score: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "reflection-agent"}

@app.get("/metrics")
async def metrics():
    return generate_latest()

@app.post("/reflect", response_model=ReflectionResponse)
@reflection_duration.time()
async def reflect_on_response(request: ReflectionRequest):
    """
    Analyze and reflect on a model's response to provide improvements.
    """
    reflection_requests_total.inc()
    
    try:
        start_time = time.time()
        
        # Simple reflection logic - in a real implementation, this would use a more sophisticated approach
        reflection = f"Analyzing response to: {request.query}"
        improvements = [
            "Consider providing more specific examples",
            "Add relevant context from the conversation history",
            "Verify factual accuracy of the response"
        ]
        confidence_score = 0.85
        
        duration = time.time() - start_time
        logger.info(f"Reflection completed in {duration:.2f}s")
        
        return ReflectionResponse(
            reflection=reflection,
            improvements=improvements,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        reflection_errors_total.inc()
        logger.error(f"Reflection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reflection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5012)
