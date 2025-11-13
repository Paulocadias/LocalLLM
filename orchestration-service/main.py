from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, TypedDict
import requests
import json
import redis
import os
import logging
from datetime import datetime
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Orchestration Service", version="1.0.0")

# Initialize Redis for state management
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Service URLs
MULTIMODAL_SERVICE_URL = os.getenv("MULTIMODAL_SERVICE_URL", "http://multimodal-agent:8002")
LANGCHAIN_SERVICE_URL = os.getenv("LANGCHAIN_SERVICE_URL", "http://langchain-agent:8000")
LOCALLLM_SERVICE_URL = os.getenv("LOCALLLM_SERVICE_URL", "http://localllm:8080")

class WorkflowType(str, Enum):
    VISION_TO_CODE = "vision_to_code"
    DOCUMENT_AUTOMATION = "document_automation"
    QUALITY_CONTROL = "quality_control"
    DASHBOARD_CREATION = "dashboard_creation"
    LEGACY_MODERNIZATION = "legacy_modernization"

class WorkflowRequest(BaseModel):
    workflow_type: WorkflowType
    input_data: Dict[str, Any]  # Can be image_data, text, requirements, etc.
    max_iterations: int = 5
    target_language: str = "react"
    requirements: Optional[str] = None

class WorkflowState(TypedDict):
    workflow_id: str
    workflow_type: WorkflowType
    input_data: Dict[str, Any]
    visual_analysis: Optional[str]
    code_requirements: Optional[str]
    generated_code: Optional[str]
    test_results: Optional[str]
    iterations: int
    status: str
    confidence_score: float
    error_messages: List[str]
    start_time: float
    end_time: Optional[float]

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    final_output: Optional[Dict[str, Any]] = None
    iterations_completed: int
    confidence_score: float
    processing_time: float
    error_messages: List[str] = []

class VisionToCodeWorkflow:
    """Vision-to-Code workflow for converting UI designs to production code"""
    
    def __init__(self):
        self.name = "vision_to_code"
        self.description = "Convert UI designs and screenshots to production code"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute vision-to-code workflow"""
        try:
            # Step 1: Visual Analysis with Qwen2.5-VL
            state = await self._vision_analysis(state)
            if state["status"] == "error":
                return state
            
            # Step 2: Code Generation with Qwen3-Coder
            state = await self._code_generation(state)
            if state["status"] == "error":
                return state
            
            # Step 3: Reflection and Testing
            state = await self._reflection_testing(state)
            
            # Step 4: Iterative Refinement
            while state["iterations"] < state["max_iterations"] and state["status"] != "completed":
                if state["test_results"] == "passed":
                    state["status"] = "completed"
                    break
                
                state["iterations"] += 1
                state = await self._refine_code(state)
            
            state["end_time"] = time.time()
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error_messages"].append(f"Workflow execution failed: {str(e)}")
            return state
    
    async def _vision_analysis(self, state: WorkflowState) -> WorkflowState:
        """Analyze image and extract UI components"""
        try:
            image_data = state["input_data"].get("image_data")
            if not image_data:
                state["status"] = "error"
                state["error_messages"].append("No image data provided")
                return state
            
            # Call multimodal service for visual analysis
            analysis_request = {
                "message": "Analyze this UI design and extract all components, layout, and functionality",
                "image_data": image_data,
                "conversation_id": state["workflow_id"]
            }
            
            response = requests.post(
                f"{MULTIMODAL_SERVICE_URL}/analyze-image",
                json=analysis_request,
                timeout=60
            )
            
            if response.status_code == 200:
                analysis_result = response.json()
                state["visual_analysis"] = analysis_result["response"]
                state["confidence_score"] = analysis_result["confidence_score"]
                logger.info(f"Visual analysis completed for workflow {state['workflow_id']}")
            else:
                state["status"] = "error"
                state["error_messages"].append(f"Visual analysis failed: {response.status_code}")
            
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error_messages"].append(f"Visual analysis error: {str(e)}")
            return state
    
    async def _code_generation(self, state: WorkflowState) -> WorkflowState:
        """Generate code from visual analysis"""
        try:
            if not state["visual_analysis"]:
                state["status"] = "error"
                state["error_messages"].append("No visual analysis available")
                return state
            
            # Prepare code generation prompt
            code_prompt = f"""
            Based on this UI analysis, generate {state["target_language"]} code:
            
            UI Analysis: {state["visual_analysis"]}
            
            Requirements: {state["requirements"] or "Create production-ready, responsive code"}
            
            Please provide:
            1. Complete, runnable code
            2. Comments explaining the implementation
            3. Any necessary dependencies
            4. Responsive design considerations
            """
            
            # Call LangChain service for code generation
            code_request = {
                "message": code_prompt,
                "conversation_id": state["workflow_id"],
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{LANGCHAIN_SERVICE_URL}/chat",
                json=code_request,
                timeout=120
            )
            
            if response.status_code == 200:
                code_result = response.json()
                state["generated_code"] = code_result["response"]
                logger.info(f"Code generation completed for workflow {state['workflow_id']}")
            else:
                state["status"] = "error"
                state["error_messages"].append(f"Code generation failed: {response.status_code}")
            
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error_messages"].append(f"Code generation error: {str(e)}")
            return state
    
    async def _reflection_testing(self, state: WorkflowState) -> WorkflowState:
        """Test and validate generated code"""
        try:
            if not state["generated_code"]:
                state["status"] = "error"
                state["error_messages"].append("No generated code available")
                return state
            
            # Simple code validation (in production, this would run actual tests)
            code_validation = self._validate_code(state["generated_code"])
            
            if code_validation["valid"]:
                state["test_results"] = "passed"
                state["confidence_score"] = max(state["confidence_score"], 0.9)
            else:
                state["test_results"] = code_validation["errors"]
                state["confidence_score"] = state["confidence_score"] * 0.8
            
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error_messages"].append(f"Reflection testing error: {str(e)}")
            return state
    
    async def _refine_code(self, state: WorkflowState) -> WorkflowState:
        """Refine code based on test results"""
        try:
            refinement_prompt = f"""
            The previous code failed validation. Please fix these issues:
            
            Original Requirements: {state["requirements"]}
            Previous Code: {state["generated_code"]}
            Test Errors: {state["test_results"]}
            
            Generate improved code that addresses these issues.
            """
            
            refinement_request = {
                "message": refinement_prompt,
                "conversation_id": state["workflow_id"],
                "temperature": 0.2
            }
            
            response = requests.post(
                f"{LANGCHAIN_SERVICE_URL}/chat",
                json=refinement_request,
                timeout=120
            )
            
            if response.status_code == 200:
                refinement_result = response.json()
                state["generated_code"] = refinement_result["response"]
                logger.info(f"Code refinement iteration {state['iterations']} completed")
                
                # Re-test the refined code
                state = await self._reflection_testing(state)
            else:
                state["status"] = "error"
                state["error_messages"].append(f"Code refinement failed: {response.status_code}")
            
            return state
            
        except Exception as e:
            state["status"] = "error"
            state["error_messages"].append(f"Code refinement error: {str(e)}")
            return state
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Simple code validation (placeholder for actual testing)"""
        # In production, this would run actual unit tests or syntax validation
        validation = {
            "valid": True,
            "errors": []
        }
        
        # Basic syntax checks
        if "error" in code.lower():
            validation["valid"] = False
            validation["errors"].append("Code contains error references")
        
        if len(code.strip()) < 50:
            validation["valid"] = False
            validation["errors"].append("Code appears incomplete")
        
        return validation

class DocumentAutomationWorkflow:
    """Document automation workflow for technical documentation"""
    
    def __init__(self):
        self.name = "document_automation"
        self.description = "Parse technical documents and generate integration code"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute document automation workflow"""
        # Implementation similar to VisionToCodeWorkflow
        # Would include document parsing, API generation, and integration testing
        state["status"] = "completed"
        state["final_output"] = {"message": "Document automation workflow - implementation pending"}
        return state

class QualityControlWorkflow:
    """Quality control workflow for visual inspection"""
    
    def __init__(self):
        self.name = "quality_control"
        self.description = "Visual inspection with defect detection and algorithm generation"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute quality control workflow"""
        # Implementation for defect detection and algorithm generation
        state["status"] = "completed"
        state["final_output"] = {"message": "Quality control workflow - implementation pending"}
        return state

# Workflow registry
WORKFLOW_REGISTRY = {
    WorkflowType.VISION_TO_CODE: VisionToCodeWorkflow(),
    WorkflowType.DOCUMENT_AUTOMATION: DocumentAutomationWorkflow(),
    WorkflowType.QUALITY_CONTROL: QualityControlWorkflow(),
    WorkflowType.DASHBOARD_CREATION: VisionToCodeWorkflow(),  # Reuse for now
    WorkflowType.LEGACY_MODERNIZATION: VisionToCodeWorkflow(),  # Reuse for now
}

@app.post("/execute-workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """Execute a multi-agent workflow"""
    try:
        workflow_id = f"workflow_{int(time.time())}_{hash(str(request.dict()))}"
        
        # Initialize workflow state
        state: WorkflowState = {
            "workflow_id": workflow_id,
            "workflow_type": request.workflow_type,
            "input_data": request.input_data,
            "visual_analysis": None,
            "code_requirements": None,
            "generated_code": None,
            "test_results": None,
            "iterations": 1,
            "status": "running",
            "confidence_score": 0.0,
            "error_messages": [],
            "start_time": time.time(),
            "end_time": None,
            "max_iterations": request.max_iterations,
            "target_language": request.target_language,
            "requirements": request.requirements
        }
        
        # Get the appropriate workflow
        workflow = WORKFLOW_REGISTRY.get(request.workflow_type)
        if not workflow:
            raise HTTPException(status_code=400, detail=f"Unknown workflow type: {request.workflow_type}")
        
        # Execute workflow
        state = await workflow.execute(state)
        
        # Prepare response
        processing_time = (state.get("end_time") or time.time()) - state["start_time"]
        
        response = WorkflowResponse(
            workflow_id=workflow_id,
            status=state["status"],
            final_output={
                "generated_code": state.get("generated_code"),
                "visual_analysis": state.get("visual_analysis"),
                "confidence_score": state["confidence_score"]
            },
            iterations_completed=state["iterations"],
            confidence_score=state["confidence_score"],
            processing_time=processing_time,
            error_messages=state["error_messages"]
        )
        
        # Store workflow result in Redis
        redis_client.setex(
            f"workflow:{workflow_id}",
            86400,  # 24 hours
            json.dumps(response.dict())
        )
        
        logger.info(f"Workflow {workflow_id} completed with status: {state['status']}")
        return response
        
    except Exception as e:
        logger.error(f"Workflow execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/{workflow_id}")
async def get_workflow_result(workflow_id: str):
    """Get workflow result by ID"""
    cached_result = redis_client.get(f"workflow:{workflow_id}")
    if cached_result:
        return json.loads(cached_result)
    else:
        raise HTTPException(status_code=404, detail="Workflow not found")

@app.get("/workflows")
async def list_workflows():
    """List available workflows"""
    return {
        "workflows": [
            {
                "type": workflow_type.value,
                "name": workflow.name,
                "description": workflow.description
            }
            for workflow_type, workflow in WORKFLOW_REGISTRY.items()
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check dependencies
    services_healthy = {}
    for service_name, service_url in [
        ("multimodal", MULTIMODAL_SERVICE_URL),
        ("langchain", LANGCHAIN_SERVICE_URL),
        ("localllm", LOCALLLM_SERVICE_URL)
    ]:
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            services_healthy[service_name] = response.status_code == 200
        except:
            services_healthy[service_name] = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": services_healthy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
