from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import redis
import os
import base64
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Agent Service", version="1.0.0")

# Initialize Redis for caching
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Initialize Ollama for multimodal models
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
multimodal_model = os.getenv("MULTIMODAL_MODEL", "qwen2.5-vl:7b")

class MultimodalRequest(BaseModel):
    message: str
    image_data: Optional[str] = None  # base64 encoded image
    conversation_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000

class MultimodalResponse(BaseModel):
    response: str
    conversation_id: str
    image_analysis: Optional[str] = None
    objects_detected: List[str] = []
    confidence_score: float

class CodeFromImageRequest(BaseModel):
    image_data: str  # base64 encoded screenshot/design
    target_language: str = "html"
    requirements: Optional[str] = None

class CodeFromImageResponse(BaseModel):
    generated_code: str
    language: str
    analysis: str
    confidence: float

class DocumentAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded document
    analysis_type: str = "text_extraction"  # text_extraction, table_extraction, diagram_analysis

class DocumentAnalysisResponse(BaseModel):
    extracted_text: str
    structured_data: Optional[Dict[str, Any]] = None
    analysis_summary: str
    confidence: float

# Multimodal tools
class ImageAnalysisTool:
    def __init__(self):
        self.name = "image_analyzer"
        self.description = "Analyzes images to extract text, objects, and visual information"

    def analyze_image(self, image_data: str, prompt: str) -> str:
        """Analyze image using Qwen2.5-VL"""
        try:
            # Prepare multimodal prompt
            multimodal_prompt = f"""
            Analyze this image and respond to: {prompt}
            
            Please provide:
            1. Objects and elements detected
            2. Text content (if any)
            3. Visual characteristics
            4. Overall context and meaning
            """
            
            response = requests.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": multimodal_model,
                    "prompt": multimodal_prompt,
                    "images": [image_data],
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "Analysis failed")
            else:
                return f"Error analyzing image: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Image analysis failed: {str(e)}"

class CodeGenerationTool:
    def __init__(self):
        self.name = "code_generator"
        self.description = "Generates code from visual designs and screenshots"

    def generate_code_from_image(self, image_data: str, target_language: str, requirements: str = "") -> Dict[str, Any]:
        """Generate code from visual design using Qwen2.5-VL"""
        try:
            prompt = f"""
            Analyze this UI/design screenshot and generate {target_language} code.
            
            Requirements: {requirements}
            
            Please provide:
            1. Complete, runnable code
            2. Comments explaining the implementation
            3. Any necessary dependencies or setup instructions
            """
            
            response = requests.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": multimodal_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "options": {"temperature": 0.5}
                }
            )
            
            if response.status_code == 200:
                generated_code = response.json().get("response", "")
                return {
                    "code": generated_code,
                    "language": target_language,
                    "confidence": 0.85
                }
            else:
                return {"error": f"Code generation failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return {"error": f"Code generation failed: {str(e)}"}

class DocumentParserTool:
    def __init__(self):
        self.name = "document_parser"
        self.description = "Extracts structured data from documents, tables, and diagrams"

    def parse_document(self, image_data: str, analysis_type: str) -> Dict[str, Any]:
        """Parse document using Qwen2.5-VL"""
        try:
            if analysis_type == "table_extraction":
                prompt = "Extract all data from this table. Provide structured output with headers and rows."
            elif analysis_type == "diagram_analysis":
                prompt = "Analyze this diagram. Describe the components, relationships, and overall structure."
            else:  # text_extraction
                prompt = "Extract all text content from this document. Preserve formatting and structure."
            
            response = requests.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": multimodal_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            
            if response.status_code == 200:
                extracted_content = response.json().get("response", "")
                return {
                    "content": extracted_content,
                    "analysis_type": analysis_type,
                    "confidence": 0.8
                }
            else:
                return {"error": f"Document parsing failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Document parsing error: {e}")
            return {"error": f"Document parsing failed: {str(e)}"}

# Initialize tools
image_analyzer = ImageAnalysisTool()
code_generator = CodeGenerationTool()
document_parser = DocumentParserTool()

@app.post("/analyze-image", response_model=MultimodalResponse)
async def analyze_image_endpoint(request: MultimodalRequest):
    """Analyze image and respond to multimodal queries"""
    try:
        if not request.image_data:
            raise HTTPException(status_code=400, detail="Image data required")
        
        # Check cache
        cache_key = f"multimodal:{hash(request.message + request.image_data)}"
        cached_response = redis_client.get(cache_key)
        
        if cached_response:
            logger.info("Returning cached multimodal response")
            return MultimodalResponse(**json.loads(cached_response))
        
        # Analyze image
        analysis_result = image_analyzer.analyze_image(request.image_data, request.message)
        
        # Extract objects (simplified - in production would use object detection)
        objects_detected = self._extract_objects_from_analysis(analysis_result)
        
        result = MultimodalResponse(
            response=analysis_result,
            conversation_id=request.conversation_id or "default",
            image_analysis=analysis_result,
            objects_detected=objects_detected,
            confidence_score=0.85
        )
        
        # Cache the response
        redis_client.setex(cache_key, 3600, json.dumps(result.dict()))
        
        return result
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-code-from-image", response_model=CodeFromImageResponse)
async def generate_code_from_image_endpoint(request: CodeFromImageRequest):
    """Generate code from UI/design screenshots"""
    try:
        result = code_generator.generate_code_from_image(
            request.image_data, 
            request.target_language, 
            request.requirements or ""
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return CodeFromImageResponse(
            generated_code=result["code"],
            language=result["language"],
            analysis="Code generated from visual design",
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document_endpoint(request: DocumentAnalysisRequest):
    """Analyze documents, tables, and diagrams"""
    try:
        result = document_parser.parse_document(request.image_data, request.analysis_type)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return DocumentAnalysisResponse(
            extracted_text=result["content"],
            structured_data=self._structure_extracted_data(result["content"]),
            analysis_summary=f"Document analyzed using {request.analysis_type}",
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process image file"""
    try:
        # Read and encode image
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image_data": base64_image,
            "size": len(image_data),
            "message": "Image uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _extract_objects_from_analysis(analysis: str) -> List[str]:
    """Extract detected objects from analysis text (simplified)"""
    # In production, this would use proper object detection
    common_objects = ["person", "car", "building", "text", "table", "diagram", "button", "form", "header", "footer"]
    detected = []
    
    for obj in common_objects:
        if obj in analysis.lower():
            detected.append(obj)
    
    return detected

def _structure_extracted_data(content: str) -> Dict[str, Any]:
    """Structure extracted document data (simplified)"""
    # In production, this would use more sophisticated parsing
    lines = content.split('\n')
    structured_data = {
        "paragraphs": [line.strip() for line in lines if len(line.strip()) > 50],
        "headings": [line.strip() for line in lines if len(line.strip()) < 50 and line.strip().isupper()],
        "tables": [],
        "key_points": []
    }
    
    return structured_data

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multimodal-agent"}

@app.get("/models")
async def list_models():
    """List available multimodal models"""
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            multimodal_models = [model for model in models if "vl" in model["name"].lower() or "vision" in model["name"].lower()]
            return {"multimodal_models": multimodal_models}
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch models")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
