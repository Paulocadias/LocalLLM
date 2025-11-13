from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from auth import APIKeyValidator, create_validator_dependency
from error_handling import (
    CircuitBreaker, retry_with_backoff,
    service_unavailable_error, timeout_error, model_error, internal_error
)
from cline_utils import ClinePromptParser
from typing import Dict, Any, List, Optional, TypedDict
import requests
import json
import redis
import os
import logging
from datetime import datetime, timedelta
from enum import Enum
import time
import numpy as np
from collections import defaultdict, deque
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta-Orchestrator Service", version="1.0.0")

# Initialize Redis for learning and state management
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Initialize API Key Validator
api_validator = APIKeyValidator(redis_client)

# Initialize Cline prompt parser for AI coding assistant support
cline_parser = ClinePromptParser()

# Migrate legacy API key on startup (if present)
legacy_key = os.getenv("API_KEY")
if legacy_key:
    api_validator.migrate_legacy_api_key(legacy_key, "Meta-Orchestrator Admin Key")
else:
    logger.info("No legacy API_KEY found to migrate")

# Create authentication dependencies
verify_chat = create_validator_dependency(redis_client, ["chat", "route"])

# Available services with enhanced model integration
SERVICES = {
    "orchestration": {
        "url": os.getenv("ORCHESTRATION_URL", "http://orchestration:8003"),
        "capabilities": ["multimodal_workflows", "vision_to_code", "document_automation", "quality_control"],
        "description": "Multi-agent workflows with iterative refinement",
        "model_manager_integration": True,
        "preferred_models": ["deepseek-r1", "qwen3-coder", "multimodal"]
    },
    "multimodal": {
        "url": os.getenv("MULTIMODAL_URL", "http://multimodal-agent:8002"),
        "capabilities": ["image_analysis", "visual_understanding", "document_parsing", "object_detection"],
        "description": "Visual understanding with enhanced models",
        "model_manager_integration": True,
        "preferred_models": ["multimodal", "deepseek-r1"]
    },
    "langchain": {
        "url": os.getenv("LANGCHAIN_URL", "http://langchain-agent:8000"),
        "capabilities": ["code_generation", "problem_solving", "research", "debugging"],
        "description": "Agentic workflows with intelligent model selection",
        "model_manager_integration": True,
        "preferred_models": ["qwen3-coder", "deepseek-r1", "coder"]
    },
    "localllm": {
        "url": os.getenv("LOCALLLM_URL", "http://localllm:8080"),
        "capabilities": ["general_chat", "reasoning", "reflection", "analysis"],
        "description": "Enhanced reasoning with reflection capabilities",
        "model_manager_integration": True,
        "preferred_models": ["deepseek-r1", "qwen3-general", "general"]
    },
    "model-manager": {
        "url": os.getenv("MODEL_MANAGER_URL", "http://model-manager:8005"),
        "capabilities": ["model_selection", "performance_analysis", "reflection_analysis"],
        "description": "Intelligent model management and reflection",
        "model_manager_integration": False  # This is the model manager itself
    },
    "web-search": {
        "url": os.getenv("WEB_SEARCH_URL", "http://web-search:8006"),
        "capabilities": ["web_search", "real_time_information", "current_events", "fact_checking"],
        "description": "Real-time web search using DuckDuckGo",
        "model_manager_integration": False
    },
    "model-improvement": {
        "url": os.getenv("MODEL_IMPROVEMENT_URL", "http://model-improvement:8008"),
        "capabilities": ["model_testing", "automatic_improvement", "version_detection", "reflexion_learning"],
        "description": "Automatic model improvement and testing with reflexion learning",
        "model_manager_integration": True,
        "preferred_models": ["deepseek-r1"]  # For analysis tasks
    }
}

# LoRA Profile Specialization System
LORA_PROFILE_MODELS = {
    # Coding Profiles
    "android": "qwen-android-mobile",
    "backend": "qwen-backend",
    "frontend": "qwen-frontend",
    "bug_fixing": "qwen-bug-fixing",
    "refactoring": "qwen-refactor",
    "documentation": "qwen-documentation",
    # Business & Consulting Profiles
    "career_advisor": "qwen-career-advisor",
    "marketing_specialist": "qwen-marketing",
    "website_builder": "qwen-website"
}

PROFILE_KEYWORDS = {
    # Coding Profiles
    "android": ["kotlin", "java", "android", "activity", "fragment", "jetpack", "recyclerview", "viewmodel"],
    "backend": ["api", "server", "database", "sql", "rest", "graphql", "microservice", "endpoint", "postgres"],
    "frontend": ["react", "javascript", "typescript", "html", "css", "component", "hooks", "jsx", "tsx"],
    "bug_fixing": ["bug", "error", "debug", "exception", "fix", "troubleshoot", "crash", "stacktrace"],
    "refactoring": ["refactor", "optimize", "improve", "performance", "clean code", "efficiency", "restructure"],
    "documentation": ["document", "explain", "comment", "docstring", "readme", "guide", "tutorial", "describe"],
    # Business & Consulting Profiles
    "career_advisor": ["career", "job", "interview", "resume", "promotion", "salary", "negotiate", "hiring", "transition", "linkedin", "networking"],
    "marketing_specialist": ["marketing", "campaign", "seo", "content", "email", "conversion", "ads", "social media", "analytics", "branding", "roi"],
    "website_builder": ["website", "landing page", "webflow", "wordpress", "responsive", "mobile", "design", "css", "html", "ux", "conversion rate"]
}

def detect_coding_profile(prompt: str) -> Optional[str]:
    """
    Detect coding profile from user prompt using keyword matching
    Returns profile name or None if no clear match
    """
    prompt_lower = prompt.lower()

    # Count keyword matches for each profile
    scores = {}
    for profile, keywords in PROFILE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[profile] = score

    # Return profile with highest score if score >= 2 (minimum confidence)
    if scores:
        best_profile = max(scores, key=scores.get)
        if scores[best_profile] >= 2:
            logger.info(f"Profile detected: {best_profile} (score: {scores[best_profile]})")
            return best_profile

    return None

class TaskType(str, Enum):
    VISUAL_CODING = "visual_coding"
    CODE_GENERATION = "code_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    QUALITY_CONTROL = "quality_control"
    GENERAL_CHAT = "general_chat"
    PROBLEM_SOLVING = "problem_solving"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    WEB_SEARCH = "web_search"

class MetaRequest(BaseModel):
    message: str
    task_type: Optional[TaskType] = None
    context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class MetaResponse(BaseModel):
    response: str
    service_used: str
    confidence: float
    processing_time: float
    session_id: str
    metadata: Dict[str, Any]

class PerformanceMetrics(TypedDict):
    success_count: int
    failure_count: int
    total_requests: int
    avg_response_time: float
    avg_confidence: float
    last_updated: float

class LearningSystem:
    """Intelligent learning system for model selection and optimization"""
    
    def __init__(self):
        self.performance_data = defaultdict(lambda: {
            "success_count": 0,
            "failure_count": 0,
            "total_requests": 0,
            "response_times": deque(maxlen=100),
            "confidence_scores": deque(maxlen=100),
            "task_success_rates": defaultdict(lambda: {"success": 0, "total": 0})
        })
    
    def record_performance(self, service: str, task_type: str, success: bool, 
                         response_time: float, confidence: float):
        """Record performance data for learning"""
        data = self.performance_data[service]
        data["total_requests"] += 1
        
        if success:
            data["success_count"] += 1
        else:
            data["failure_count"] += 1
        
        data["response_times"].append(response_time)
        data["confidence_scores"].append(confidence)
        
        task_data = data["task_success_rates"][task_type]
        task_data["total"] += 1
        if success:
            task_data["success"] += 1
    
    def get_service_score(self, service: str, task_type: str) -> float:
        """Calculate intelligent score for service selection"""
        if service not in self.performance_data:
            return 0.5  # Default score for new services
        
        data = self.performance_data[service]
        
        if data["total_requests"] == 0:
            return 0.5
        
        # Calculate base success rate
        success_rate = data["success_count"] / data["total_requests"]
        
        # Task-specific success rate
        task_data = data["task_success_rates"][task_type]
        task_success_rate = task_data["success"] / task_data["total"] if task_data["total"] > 0 else success_rate
        
        # Response time factor (faster is better)
        avg_response_time = np.mean(data["response_times"]) if data["response_times"] else 10.0
        response_factor = max(0, 1 - (avg_response_time / 30.0))  # Normalize to 30s max
        
        # Confidence factor
        avg_confidence = np.mean(data["confidence_scores"]) if data["confidence_scores"] else 0.5
        
        # Combined score with weights
        score = (
            0.4 * task_success_rate +
            0.3 * success_rate +
            0.2 * response_factor +
            0.1 * avg_confidence
        )
        
        return min(1.0, max(0.1, score))
    
    def get_best_service(self, task_type: str, capabilities_needed: List[str]) -> str:
        """Select the best service for a given task"""
        candidate_services = []
        
        for service_name, service_info in SERVICES.items():
            # Check if service has required capabilities
            if any(cap in service_info["capabilities"] for cap in capabilities_needed):
                score = self.get_service_score(service_name, task_type)
                candidate_services.append((service_name, score, service_info))
        
        if not candidate_services:
            # Fallback to most capable service
            return "localllm"
        
        # Sort by score and select best
        candidate_services.sort(key=lambda x: x[1], reverse=True)
        best_service = candidate_services[0][0]
        
        logger.info(f"Selected service '{best_service}' for task '{task_type}' with score {candidate_services[0][1]:.3f}")
        return best_service

class TaskClassifier:
    """Intelligent task classification system"""
    
    def __init__(self):
        self.keyword_patterns = {
            TaskType.VISUAL_CODING: ["screenshot", "ui design", "convert to code", "html", "css", "react", "component"],
            TaskType.CODE_GENERATION: ["write code", "function", "class", "algorithm", "implement", "python", "javascript"],
            TaskType.DOCUMENT_ANALYSIS: ["document", "pdf", "extract", "parse", "table", "diagram", "manual"],
            TaskType.QUALITY_CONTROL: ["defect", "inspect", "quality", "detect", "analyze image", "find issues"],
            TaskType.PROBLEM_SOLVING: ["solve", "problem", "issue", "debug", "fix", "troubleshoot"],
            TaskType.RESEARCH: ["research", "explain", "what is", "how does", "compare", "analysis"],
            TaskType.DEBUGGING: ["error", "bug", "not working", "fix code", "debug", "exception"],
            TaskType.WEB_SEARCH: ["search the web", "search for", "find online", "latest news", "current", "recent", "what's happening", "look up"]
        }
    
    def classify_task(self, message: str, context: Optional[Dict[str, Any]] = None) -> TaskType:
        """Classify task type based on message content and context"""
        message_lower = message.lower()
        
        # Check for explicit task type in context
        if context and context.get("task_type"):
            try:
                return TaskType(context["task_type"])
            except:
                pass
        
        # Keyword-based classification
        scores = defaultdict(int)
        
        for task_type, keywords in self.keyword_patterns.items():
            for keyword in keywords:
                if keyword in message_lower:
                    scores[task_type] += 1
        
        if scores:
            best_task = max(scores.items(), key=lambda x: x[1])[0]
            return best_task
        
        # Default to general chat
        return TaskType.GENERAL_CHAT
    
    def get_required_capabilities(self, task_type: TaskType) -> List[str]:
        """Get required capabilities for a task type"""
        capability_map = {
            TaskType.VISUAL_CODING: ["vision_to_code", "image_analysis", "code_generation"],
            TaskType.CODE_GENERATION: ["code_generation", "problem_solving"],
            TaskType.DOCUMENT_ANALYSIS: ["document_parsing", "visual_understanding"],
            TaskType.QUALITY_CONTROL: ["object_detection", "image_analysis"],
            TaskType.PROBLEM_SOLVING: ["problem_solving", "reasoning"],
            TaskType.RESEARCH: ["research", "analysis"],
            TaskType.DEBUGGING: ["debugging", "problem_solving"],
            TaskType.GENERAL_CHAT: ["general_chat", "reasoning"],
            TaskType.WEB_SEARCH: ["web_search", "real_time_information"]
        }

        return capability_map.get(task_type, ["general_chat"])

class ModelManagerIntegration:
    """Integration with model manager for intelligent model selection"""
    
    def __init__(self):
        self.model_manager_url = SERVICES["model-manager"]["url"]
    
    async def get_recommended_model(self, task_type: str, context: str = "") -> Dict[str, Any]:
        """Get recommended model from model manager"""
        try:
            response = requests.get(
                f"{self.model_manager_url}/recommend-model",
                params={"task_type": task_type, "context": context},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Model manager returned status {response.status_code}")
                return self._get_fallback_model(task_type)
        except Exception as e:
            logger.error(f"Error calling model manager: {str(e)}")
            return self._get_fallback_model(task_type)
    
    def _get_fallback_model(self, task_type: str) -> Dict[str, Any]:
        """Fallback model selection when model manager is unavailable"""
        fallback_map = {
            "coding": {"model_type": "coder", "model_name": "qwen2.5-coder:7b"},
            "programming": {"model_type": "coder", "model_name": "qwen2.5-coder:7b"},
            "multimodal": {"model_type": "multimodal", "model_name": "qwen2.5-vl:7b"},
            "vision": {"model_type": "multimodal", "model_name": "qwen2.5-vl:7b"},
            "reasoning": {"model_type": "deepseek-r1", "model_name": "deepseek-r1:7b"},
            "mathematics": {"model_type": "deepseek-r1", "model_name": "deepseek-r1:7b"},
            "general": {"model_type": "general", "model_name": "qwen2.5:7b"},
            "chat": {"model_type": "general", "model_name": "qwen2.5:7b"}
        }
        
        task_lower = task_type.lower()
        for key, model in fallback_map.items():
            if key in task_lower:
                return {
                    "task_type": task_type,
                    "recommended_model_type": model["model_type"],
                    "recommended_model": model["model_name"],
                    "score": 0.7,
                    "description": f"Fallback model for {task_type}",
                    "reasoning": "Using fallback model selection",
                    "reflection_enabled": False
                }
        
        return {
            "task_type": task_type,
            "recommended_model_type": "general",
            "recommended_model": "qwen2.5:7b",
            "score": 0.5,
            "description": "General fallback model",
            "reasoning": "Using general fallback model",
            "reflection_enabled": False
        }
    
    async def analyze_model_performance(self, model_type: str, task_results: str = "") -> Dict[str, Any]:
        """Analyze model performance using reflection"""
        try:
            response = requests.get(
                f"{self.model_manager_url}/models/reflection/analyze-performance",
                params={"model_type": model_type, "task_results": task_results},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Model manager reflection unavailable"}
        except Exception as e:
            logger.error(f"Error calling model reflection: {str(e)}")
            return {"error": str(e)}

# Initialize systems
learning_system = LearningSystem()
task_classifier = TaskClassifier()
model_manager = ModelManagerIntegration()

# Initialize circuit breakers for each service
service_circuit_breakers = {
    service_name: CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30,
        expected_exception=requests.exceptions.RequestException
    )
    for service_name in SERVICES.keys()
}
logger.info(f"Initialized circuit breakers for {len(service_circuit_breakers)} services")

@app.post("/chat", response_model=MetaResponse)
async def intelligent_chat(request: MetaRequest, auth: dict = Depends(verify_chat)):
    """Intelligent chat endpoint that learns and selects best service (requires authentication)"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Step 1: Classify the task
        task_type = task_classifier.classify_task(request.message, request.context)
        logger.info(f"Classified task as: {task_type.value}")
        
        # Step 2: Determine required capabilities
        capabilities_needed = task_classifier.get_required_capabilities(task_type)
        
        # Step 3: Select best service using learning system
        best_service = learning_system.get_best_service(task_type.value, capabilities_needed)
        service_info = SERVICES[best_service]
        
        # Step 4: Route request to selected service
        if best_service == "orchestration":
            # Use orchestration for complex workflows
            workflow_type = _determine_workflow_type(task_type)
            workflow_request = {
                "workflow_type": workflow_type,
                "input_data": {
                    "message": request.message,
                    "context": request.context or {}
                },
                "requirements": request.preferences or {}
            }
            response = requests.post(
                f"{service_info['url']}/execute-workflow",
                json=workflow_request,
                timeout=300
            )
        else:
            # Use direct service call
            chat_request = {
                "message": request.message,
                "conversation_id": session_id,
                "model": "qwen3:latest",
                "context": request.context or {}
            }
            headers = {}
            # Add auth header if calling localllm (requires Bearer token)
            if best_service == "localllm":
                api_key = os.getenv("LOCALLLM_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="LOCALLLM_API_KEY not configured")
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.post(
                f"{service_info['url']}/chat",
                json=chat_request,
                headers=headers,
                timeout=120
            )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract confidence from response
            confidence = result.get("confidence_score", 0.8)
            if "confidence" in result:
                confidence = result["confidence"]
            
            # Record successful performance
            learning_system.record_performance(
                best_service, task_type.value, True, 
                processing_time, confidence
            )
            
            return MetaResponse(
                response=result.get("response", "No response generated"),
                service_used=best_service,
                confidence=confidence,
                processing_time=processing_time,
                session_id=session_id,
                metadata={
                    "task_type": task_type.value,
                    "capabilities_used": capabilities_needed,
                    "service_description": service_info["description"]
                }
            )
        else:
            # Record failed performance
            learning_system.record_performance(
                best_service, task_type.value, False, 
                processing_time, 0.0
            )
            
            # Fallback to local LLM
            return await _fallback_to_localllm(request, session_id, start_time)
            
    except Exception as e:
        logger.error(f"Meta-orchestrator error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"Request details: service={best_service}, url={service_info['url']}, request={chat_request}")
        processing_time = time.time() - start_time
        return await _fallback_to_localllm(request, session_id, start_time)

async def _fallback_to_localllm(request: MetaRequest, session_id: str, start_time: float) -> MetaResponse:
    """Fallback to local LLM when other services fail"""
    try:
        chat_request = {
            "message": request.message,
            "conversation_id": session_id,
            "model": "qwen3:latest",
            "context": request.context or {}
        }
        api_key = os.getenv("LOCALLLM_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="LOCALLLM_API_KEY not configured")
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.post(
            f"{SERVICES['localllm']['url']}/chat",
            json=chat_request,
            headers=headers,
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return MetaResponse(
                response=result.get("response", "Fallback response"),
                service_used="localllm",
                confidence=0.5,  # Lower confidence for fallback
                processing_time=processing_time,
                session_id=session_id,
                metadata={"fallback": True, "reason": "Service failure"}
            )
        else:
            logger.error(f"Fallback to localllm failed with status {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail="All services unavailable")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Fallback error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"Fallback request details: url={SERVICES['localllm']['url']}/chat, request={chat_request}")
        raise HTTPException(status_code=500, detail="System temporarily unavailable")

def _determine_workflow_type(task_type: TaskType) -> str:
    """Determine appropriate workflow type for orchestration"""
    workflow_map = {
        TaskType.VISUAL_CODING: "vision_to_code",
        TaskType.DOCUMENT_ANALYSIS: "document_automation", 
        TaskType.QUALITY_CONTROL: "quality_control",
        TaskType.CODE_GENERATION: "legacy_modernization"
    }
    return workflow_map.get(task_type, "vision_to_code")

@app.post("/route")
async def route_task(request: MetaRequest, auth: dict = Depends(verify_chat)):
    """
    Route endpoint - returns routing decision without executing (requires authentication)
    Returns which service would handle the task and why
    """
    try:
        # Classify the task
        task_type = task_classifier.classify_task(request.message, request.context)
        logger.info(f"Route request - classified as: {task_type.value}")

        # Determine required capabilities
        capabilities_needed = task_classifier.get_required_capabilities(task_type)

        # Get best service using learning system
        best_service = learning_system.get_best_service(task_type.value, capabilities_needed)
        service_info = SERVICES[best_service]
        service_score = learning_system.get_service_score(best_service, task_type.value)

        return {
            "task_type": task_type.value,
            "selected_service": best_service,
            "service_url": service_info["url"],
            "service_description": service_info["description"],
            "capabilities_needed": capabilities_needed,
            "service_capabilities": service_info["capabilities"],
            "confidence_score": round(service_score, 3),
            "message": f"Task '{request.message[:50]}...' would be routed to {best_service}"
        }
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for all services"""
    metrics = {}
    for service_name in SERVICES.keys():
        score = learning_system.get_service_score(service_name, "general_chat")
        data = learning_system.performance_data[service_name]
        
        metrics[service_name] = {
            "score": score,
            "success_rate": data["success_count"] / max(1, data["total_requests"]),
            "total_requests": data["total_requests"],
            "avg_response_time": np.mean(data["response_times"]) if data["response_times"] else 0,
            "task_success_rates": dict(data["task_success_rates"])
        }
    
    return metrics

@app.get("/services")
async def list_services():
    """List available services with capabilities"""
    return {
        "services": [
            {
                "name": name,
                "capabilities": info["capabilities"],
                "description": info["description"],
                "url": info["url"]
            }
            for name, info in SERVICES.items()
        ]
    }

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: dict, auth: dict = Depends(verify_chat)):
    """
    OpenAI-compatible chat completions endpoint for Cline and other tools
    Wraps meta-orchestrator intelligence in OpenAI format
    """
    try:
        # Extract message from OpenAI format
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get last user message and extract text
        user_msg = next((msg for msg in reversed(messages) if msg["role"] == "user"), None)
        if not user_msg:
            raise HTTPException(status_code=400, detail="No user message found")

        # Handle both string and structured content formats
        content = user_msg["content"]
        if isinstance(content, list):
            # Structured content (AI assistant format): [{"type": "text", "text": "..."}]
            user_message = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        elif isinstance(content, str):
            # Simple string content (standard OpenAI format)
            user_message = content
        else:
            raise HTTPException(status_code=400, detail="Invalid content format")

        if not user_message:
            raise HTTPException(status_code=400, detail="No text content found")

        # Parse Cline prompts to extract clean user query
        # Keeps all enhancements active while removing verbose internal prompts
        if cline_parser.is_cline_prompt(user_message):
            logger.info("Detected Cline prompt, extracting clean user query [PRIVACY MODE]")
            parsed_query = cline_parser.extract_user_query(user_message)
            user_message = parsed_query if parsed_query else user_message
            logger.info(f"Query extracted [length: {len(user_message)} chars, details not logged for privacy]")

        # Call internal chat endpoint with meta-orchestrator logic
        meta_request = MetaRequest(message=user_message, context={})
        meta_response = await intelligent_chat(meta_request, auth)

        # Convert to OpenAI format
        return {
            "id": f"chatcmpl-{meta_response.session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "meta-orchestrator"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": meta_response.response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(meta_response.response.split()),
                "total_tokens": len(user_message.split()) + len(meta_response.response.split())
            },
            "x_meta_orchestrator": {
                "service_used": meta_response.service_used,
                "confidence": meta_response.confidence,
                "processing_time": meta_response.processing_time,
                "task_type": meta_response.metadata.get("task_type", "unknown") if meta_response.metadata else "unknown"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_healthy = {}
    for service_name, service_info in SERVICES.items():
        try:
            response = requests.get(f"{service_info['url']}/health", timeout=5)
            services_healthy[service_name] = response.status_code == 200
        except:
            services_healthy[service_name] = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": services_healthy,
        "learning_system": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
