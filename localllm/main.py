import asyncio
import logging
import os
import time
import secrets
import json
from typing import Dict, Any, List, Optional
import schedule
import threading
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from prometheus_client.core import CollectorRegistry
import requests
import sys
sys.path.append("/app")
from localllm.cache_utils import PromptCacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enhanced Local LLM", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True
)

# Vector DB configuration
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://vector-db:8007")

# Clear any existing metrics to avoid duplication
for collector in list(REGISTRY._collector_to_names.keys()):
    REGISTRY.unregister(collector)

# Prometheus metrics
REQUEST_COUNT = Counter('localllm_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('localllm_request_duration_seconds', 'Request duration')
MODEL_UPDATE_COUNT = Counter('localllm_model_updates_total', 'Total model updates')

# Request models
class ChatRequest(BaseModel):
    message: str
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.7
    max_tokens: int = 2048
    conversation_id: Optional[str] = None
    use_rag: bool = False  # Enable RAG (Retrieval Augmented Generation)
    rag_top_k: int = 3  # Number of documents to retrieve
    rag_collection: str = "knowledge_base"  # Vector DB collection to search

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    processing_time: float

class ModelInfo(BaseModel):
    name: str
    size: str = "unknown"
    modified: str = "unknown"
    status: str = "available"

class ReflectRequest(BaseModel):
    query: str
    context: str
    model_response: str
    model: str = "qwen3:latest"

class ReflectResponse(BaseModel):
    analysis: Dict[str, Any]
    confidence: float
    suggestions: List[str]
    improved_response: str = None

# RAG (Retrieval Augmented Generation) helper functions
async def retrieve_context(query: str, collection: str = "knowledge_base", top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant context from vector database"""
    try:
        response = requests.post(
            f"{VECTOR_DB_URL}/search",
            json={
                "query": query,
                "collection_name": collection,
                "top_k": top_k
            },
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            logger.warning(f"Vector DB search failed with status {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error retrieving context from vector DB: {e}")
        return []

def augment_prompt_with_context(original_prompt: str, context_results: List[Dict[str, Any]]) -> str:
    """Augment user prompt with retrieved context"""
    if not context_results:
        return original_prompt

    context_text = "\n\n".join([
        f"[Context {i+1}]: {result['content']}"
        for i, result in enumerate(context_results)
    ])

    augmented_prompt = f"""Based on the following context information, please answer the user's question.

Context:
{context_text}

User Question: {original_prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain relevant information, answer based on your general knowledge but mention that the information wasn't found in the context."""

    return augmented_prompt

# Enhanced LLM Service
class EnhancedLLMService:
    def __init__(self):
        self.available_models = {}
        self.current_model = "qwen2.5-coder:7b"
        self.cache_manager = PromptCacheManager(redis_client, cache_ttl=3600)
        self.load_models()
        
    def load_models(self):
        """Load available models from Ollama"""
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                # Convert size to string to avoid Pydantic validation errors
                for model in data.get("models", []):
                    if "size" in model and isinstance(model["size"], int):
                        model["size"] = str(model["size"])
                    elif "size" not in model:
                        model["size"] = "unknown"
                    # Ensure all required fields exist
                    if "modified_at" not in model:
                        model["modified_at"] = "unknown"
                self.available_models = {model["name"]: model for model in data.get("models", [])}
                logger.info(f"Loaded {len(self.available_models)} models from {ollama_url}")
                # Debug: log what we actually have
                for name, model in self.available_models.items():
                    logger.info(f"Model {name}: size={model.get('size')}, type={type(model.get('size'))}")
        except Exception as e:
            logger.warning(f"Failed to load models from Ollama: {e}")
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Enhanced chat with reflection and reasoning"""
        start_time = time.time()

        # Check intelligent cache first
        cached = self.cache_manager.get_cached_response(
            prompt=request.message,
            model=request.model,
            temperature=request.temperature
        )
        if cached:
            return ChatResponse(
                response=cached["response"],
                model=cached["model"],
                tokens_used=0,
                processing_time=time.time() - start_time
            )

        # RAG: Retrieve context from vector database if enabled
        prompt_to_use = request.message
        if request.use_rag:
            try:
                context_results = await retrieve_context(
                    query=request.message,
                    collection=request.rag_collection,
                    top_k=request.rag_top_k
                )
                if context_results:
                    prompt_to_use = augment_prompt_with_context(request.message, context_results)
                    logger.info(f"RAG: Retrieved {len(context_results)} context documents for query")
                else:
                    logger.info("RAG: No context found, using original prompt")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}, falling back to original prompt")
                prompt_to_use = request.message

        # Generate response using Ollama
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_request = {
                "model": request.model,
                "prompt": prompt_to_use,  # Use augmented prompt if RAG is enabled
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=ollama_request)
            if response.status_code == 200:
                data = response.json()

                # Analyze conversation history if conversation_id is provided
                conversation_analysis = None
                force_reflection = False  # Flag to trigger reflection based on conversation context
                if request.conversation_id:
                    conversation_analysis = analyze_conversation_history(request.conversation_id)
                    if conversation_analysis.get("session_found"):
                        # Trigger forced reflection if conversation shows quality issues
                        quality_score = conversation_analysis.get("quality_score", 1.0)
                        opportunities = conversation_analysis.get("improvement_opportunities", [])

                        # Force reflection if quality is low or negative feedback detected
                        if quality_score < 0.7 or any(o.get("type") == "negative_feedback" for o in opportunities):
                            force_reflection = True
                            logger.info(f"Conversation analysis triggered enhanced reflection (quality: {quality_score})")

                # Apply Chain-of-Verification reflection and enhancement
                enhanced_response, reflection_metadata = await self._apply_reflection(
                    data["response"],
                    request.message,
                    conversation_context=conversation_analysis,
                    force_deep_reflection=force_reflection
                )

                # Store reflection metadata in Redis for learning
                try:
                    reflection_key = f"localllm:reflection:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    reflection_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": request.model,
                        "question": request.message,
                        "original_response": data["response"],
                        "final_response": enhanced_response,
                        **reflection_metadata
                    }
                    redis_client.setex(reflection_key, 60 * 60 * 24 * 7, json.dumps(reflection_data))  # 7 days

                    # Update aggregated reflection stats
                    if reflection_metadata.get("confidence", 1.0) < 0.7:
                        redis_client.hincrby(f"localllm:reflection:stats:{request.model}", "low_confidence_count", 1)
                    redis_client.hincrby(f"localllm:reflection:stats:{request.model}", "total_reflections", 1)
                except Exception as e:
                    logger.error(f"Failed to store reflection metadata: {e}")

                # Cache the response using intelligent caching
                processing_time = time.time() - start_time
                self.cache_manager.cache_response(
                    prompt=request.message,
                    model=request.model,
                    response=enhanced_response,
                    processing_time=processing_time,
                    temperature=request.temperature
                )

                return ChatResponse(
                    response=enhanced_response,
                    model=request.model,
                    tokens_used=data.get("eval_count", 0),
                    processing_time=processing_time
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate response")
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def chat_stream(self, request: ChatRequest):
        """Stream chat responses using Server-Sent Events"""
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_request = {
                "model": request.model,
                "prompt": request.message,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }

            # Make streaming request to Ollama
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=ollama_request,
                stream=True,
                timeout=120
            )

            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                token = chunk["response"]
                                full_response += token
                                # Send SSE formatted data
                                yield f"data: {json.dumps({'token': token, 'done': chunk.get('done', False)})}\n\n"

                            if chunk.get("done", False):
                                # Store conversation history if conversation_id is provided
                                if request.conversation_id:
                                    store_conversation_turn(
                                        conversation_id=request.conversation_id,
                                        user_message=request.message,
                                        ai_response=full_response,
                                        model=request.model
                                    )
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"data: {json.dumps({'error': 'Failed to generate response', 'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    async def _apply_reflection(self, response: str, original_prompt: str,
                                conversation_context: Dict[str, Any] = None,
                                force_deep_reflection: bool = False) -> tuple[str, Dict[str, Any]]:
        """
        Apply Chain-of-Verification (CoVe) reflection with multi-criteria scoring
        Now integrates conversation analysis to improve responses based on session history

        Args:
            response: Initial LLM response
            original_prompt: User's original question
            conversation_context: Optional conversation analysis results
            force_deep_reflection: Force full CoVe process regardless of quality

        Returns: (improved_response, reflection_metadata)
        """
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            # Add conversation insights to metadata
            conversation_insights = {}
            if conversation_context and conversation_context.get("session_found"):
                conversation_insights = {
                    "conversation_quality": conversation_context.get("quality_score", 1.0),
                    "detected_issues": len(conversation_context.get("improvement_opportunities", [])),
                    "force_reflection_triggered": force_deep_reflection
                }

            # Step 1: Generate verification questions (CoVe)
            verification_prompt = f"""Given this question and answer, generate 2-3 verification questions to check accuracy and completeness.

ORIGINAL QUESTION: {original_prompt}
DRAFT ANSWER: {response}

Generate ONLY the verification questions, one per line, numbered 1-3. Be concise."""

            verify_request = {
                "model": self.current_model,
                "prompt": verification_prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }

            verify_resp = requests.post(f"{ollama_url}/api/generate", json=verify_request, timeout=30)
            verification_questions = verify_resp.json().get("response", "").strip() if verify_resp.status_code == 200 else ""

            # Step 2: Multi-criteria quality scoring
            scoring_prompt = f"""Score this response on these criteria (0-10 scale):

ORIGINAL QUESTION: {original_prompt}
RESPONSE: {response}

Rate each criterion:
1. ACCURACY: Factual correctness
2. COMPLETENESS: Addresses all aspects
3. RELEVANCE: Stays on topic
4. CLARITY: Easy to understand

Return ONLY 4 numbers separated by commas (e.g., "8,7,9,8")"""

            score_request = {
                "model": self.current_model,
                "prompt": scoring_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }

            score_resp = requests.post(f"{ollama_url}/api/generate", json=score_request, timeout=30)
            scores_text = score_resp.json().get("response", "7,7,7,7").strip() if score_resp.status_code == 200 else "7,7,7,7"

            # Parse scores
            try:
                scores = [float(s.strip()) / 10.0 for s in scores_text.split(",")[:4]]
                if len(scores) < 4:
                    scores = [0.7, 0.7, 0.7, 0.7]
            except:
                scores = [0.7, 0.7, 0.7, 0.7]

            accuracy_score, completeness_score, relevance_score, clarity_score = scores

            # Calculate overall confidence (weighted average)
            confidence = (accuracy_score * 0.4 + completeness_score * 0.3 +
                         relevance_score * 0.2 + clarity_score * 0.1)

            reflection_metadata = {
                "verification_questions": verification_questions,
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "relevance": relevance_score,
                "clarity": clarity_score,
                "confidence": confidence,
                **conversation_insights  # Merge conversation analysis insights
            }

            # Step 3: Improve response if confidence is low (<0.7) OR forced by conversation analysis
            improved_response = response
            if confidence < 0.7 or force_deep_reflection:
                improvement_prompt = f"""The following response has quality issues. Improve it based on these verification questions:

ORIGINAL QUESTION: {original_prompt}
CURRENT RESPONSE: {response}

VERIFICATION QUESTIONS:
{verification_questions}

QUALITY SCORES: Accuracy={accuracy_score:.1f}/1.0, Completeness={completeness_score:.1f}/1.0

Provide an IMPROVED response that addresses the verification questions and quality gaps."""

                improve_request = {
                    "model": self.current_model,
                    "prompt": improvement_prompt,
                    "stream": False
                }

                improve_resp = requests.post(f"{ollama_url}/api/generate", json=improve_request, timeout=45)
                if improve_resp.status_code == 200:
                    improved_response = improve_resp.json()["response"]
                    reflection_metadata["improved"] = True
                    logger.info(f"Response improved via CoVe (confidence: {confidence:.2f})")

            return improved_response, reflection_metadata

        except Exception as e:
            logger.warning(f"Advanced reflection failed: {e}")
            # Return original with default metadata
            return response, {
                "accuracy": 0.7, "completeness": 0.7, "relevance": 0.7, "clarity": 0.7,
                "confidence": 0.7, "error": str(e)
            }
    
    def check_model_updates(self):
        """Check for model updates and download new versions"""
        try:
            # Get list of available models from Ollama library
            response = requests.get("https://ollama.com/library")
            if response.status_code == 200:
                # This is a simplified check - in production you'd parse the actual model list
                MODEL_UPDATE_COUNT.inc()
                logger.info("Model update check completed")
        except Exception as e:
            logger.warning(f"Model update check failed: {e}")

# API Key Management
security = HTTPBearer()

# Generate a secure API key if none exists
def generate_api_key():
    api_key = os.getenv("LOCALLLM_API_KEY")
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        logger.info(f"Generated new API key: {api_key}")
        # Store in Redis for persistence
        redis_client.set("localllm:api_key", api_key)
    return api_key

# Verify API key
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    stored_key = redis_client.get("localllm:api_key")
    if not stored_key:
        # Generate initial key
        stored_key = generate_api_key()
    
    if credentials.credentials != stored_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Session History Management
def store_conversation_turn(conversation_id: str, user_message: str, ai_response: str, model: str):
    """Store a conversation turn in Redis for history tracking and analysis"""
    try:
        session_key = f"localllm:session:{conversation_id}"
        turn_data = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "model": model
        }

        # Append to session history (using Redis list)
        redis_client.rpush(session_key, json.dumps(turn_data))

        # Set expiration (7 days)
        redis_client.expire(session_key, 60 * 60 * 24 * 7)

        logger.info(f"Stored conversation turn for session {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")

def analyze_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """Analyze conversation history to detect improvement opportunities"""
    try:
        session_key = f"localllm:session:{conversation_id}"
        history_raw = redis_client.lrange(session_key, 0, -1)

        if not history_raw:
            return {"improvement_opportunities": [], "session_found": False}

        history = [json.loads(turn) for turn in history_raw]
        opportunities = []

        # Pattern detection keywords
        negative_feedback = ["no", "wrong", "incorrect", "not what", "try again", "that's not",
                           "didn't work", "not right", "mistake", "error in"]
        clarification = ["i mean", "let me clarify", "what i meant", "actually", "to be clear",
                        "more specifically", "in other words"]
        repetition_phrases = ["again", "as i said", "i already asked", "repeat", "once more"]

        for i in range(1, len(history)):
            prev_turn = history[i-1]
            curr_turn = history[i]

            user_msg = curr_turn["user_message"].lower()
            prev_user_msg = prev_turn["user_message"].lower()

            opportunity = None

            # Detect negative feedback
            if any(phrase in user_msg for phrase in negative_feedback):
                opportunity = {
                    "type": "negative_feedback",
                    "turn_index": i,
                    "timestamp": curr_turn["timestamp"],
                    "previous_response": prev_turn["ai_response"],
                    "user_followup": curr_turn["user_message"],
                    "severity": "high",
                    "description": "User expressed dissatisfaction with previous response"
                }

            # Detect clarification attempts
            elif any(phrase in user_msg for phrase in clarification):
                opportunity = {
                    "type": "clarification",
                    "turn_index": i,
                    "timestamp": curr_turn["timestamp"],
                    "previous_response": prev_turn["ai_response"],
                    "user_followup": curr_turn["user_message"],
                    "severity": "medium",
                    "description": "User needed to clarify their question (initial response may have been off-target)"
                }

            # Detect question repetition
            elif any(phrase in user_msg for phrase in repetition_phrases):
                opportunity = {
                    "type": "repetition",
                    "turn_index": i,
                    "timestamp": curr_turn["timestamp"],
                    "previous_response": prev_turn["ai_response"],
                    "user_followup": curr_turn["user_message"],
                    "severity": "medium",
                    "description": "User asked same/similar question again (initial response incomplete)"
                }

            # Detect similar questions (semantic similarity check)
            elif len(user_msg) > 10 and len(prev_user_msg) > 10:
                # Simple word overlap check
                user_words = set(user_msg.split())
                prev_words = set(prev_user_msg.split())
                common_words = user_words & prev_words - {"the", "a", "an", "is", "are", "what", "how", "why", "can", "you", "i"}

                if len(common_words) >= 3:  # At least 3 meaningful common words
                    opportunity = {
                        "type": "rephrasing",
                        "turn_index": i,
                        "timestamp": curr_turn["timestamp"],
                        "previous_response": prev_turn["ai_response"],
                        "user_followup": curr_turn["user_message"],
                        "severity": "low",
                        "description": "User rephrased similar question (initial response may not have fully addressed the query)"
                    }

            if opportunity:
                opportunities.append(opportunity)
                # Store each opportunity for learning
                # Get model from previous turn
                model = prev_turn.get("model", "unknown")
                store_improvement_opportunity(opportunity, model)

        # Calculate session quality score
        total_turns = len(history)
        issues_count = len(opportunities)
        quality_score = max(0.0, 1.0 - (issues_count / max(total_turns, 1)))

        return {
            "conversation_id": conversation_id,
            "session_found": True,
            "total_turns": total_turns,
            "improvement_opportunities": opportunities,
            "quality_score": quality_score,
            "recommendations": [
                "Consider using reflection before responding" if issues_count > 2 else None,
                "Model may need more context or examples" if any(o["type"] == "clarification" for o in opportunities) else None,
                "Responses may be incomplete or unclear" if any(o["type"] in ["repetition", "rephrasing"] for o in opportunities) else None
            ]
        }

    except Exception as e:
        logger.error(f"Failed to analyze conversation history: {e}")
        return {"error": str(e), "session_found": False}

def store_improvement_opportunity(opportunity: Dict[str, Any], model: str):
    """Store improvement opportunity in Redis for learning and aggregation"""
    try:
        # Store individual opportunity with timestamp-based key
        opportunity_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        opportunity_key = f"localllm:learning:opportunity:{opportunity_id}"

        # Add model and storage timestamp
        enriched_opportunity = {
            **opportunity,
            "model": model,
            "stored_at": datetime.now().isoformat()
        }

        redis_client.setex(opportunity_key, 60 * 60 * 24 * 30, json.dumps(enriched_opportunity))  # 30 days

        # Update aggregated statistics by type
        type_key = f"localllm:learning:stats:{opportunity['type']}"
        redis_client.hincrby(type_key, "count", 1)
        redis_client.hincrby(type_key, f"severity_{opportunity['severity']}", 1)
        redis_client.expire(type_key, 60 * 60 * 24 * 30)

        # Update model-specific statistics
        model_stats_key = f"localllm:learning:model_stats:{model}"
        redis_client.hincrby(model_stats_key, "total_issues", 1)
        redis_client.hincrby(model_stats_key, f"type_{opportunity['type']}", 1)
        redis_client.expire(model_stats_key, 60 * 60 * 24 * 30)

        logger.info(f"Stored improvement opportunity: {opportunity['type']} (severity: {opportunity['severity']})")
    except Exception as e:
        logger.error(f"Failed to store improvement opportunity: {e}")

# Initialize service
llm_service = EnhancedLLMService()

# API Routes
@app.get("/")
async def root():
    return {"message": "Enhanced Local LLM Service", "version": "1.0.0"}

@app.get("/admin/generate-key")
async def generate_key_endpoint():
    """Generate a new API key (admin only - no auth required for initial setup)"""
    new_key = generate_api_key()
    return {
        "message": "New API key generated",
        "api_key": new_key,
        "instructions": "Use this key in the Authorization header as: Bearer <api_key>"
    }

@app.get("/admin/current-key")
async def get_current_key():
    """Get the current API key (admin only - no auth required for initial setup)"""
    current_key = redis_client.get("localllm:api_key")
    if not current_key:
        current_key = generate_api_key()
    return {
        "current_api_key": current_key,
        "instructions": "Use this key in the Authorization header as: Bearer <api_key>"
    }

@app.get("/admin/stats")
async def get_stats():
    """Get aggregated system statistics"""
    try:
        # Get request counts from Prometheus metrics
        chat_requests = REQUEST_COUNT.labels(method="POST", endpoint="/chat")._value._value
        reflect_requests = REQUEST_COUNT.labels(method="POST", endpoint="/reflect")._value._value
        models_requests = REQUEST_COUNT.labels(method="GET", endpoint="/models")._value._value

        total_requests = int(chat_requests + reflect_requests + models_requests)

        # Get Redis stats
        redis_info = redis_client.info("stats")
        keyspace_hits = int(redis_info.get("keyspace_hits", 0))
        keyspace_misses = int(redis_info.get("keyspace_misses", 0))

        # Calculate cache hit ratio
        total_cache_ops = keyspace_hits + keyspace_misses
        cache_hit_ratio = (keyspace_hits / total_cache_ops) if total_cache_ops > 0 else 0.0

        # Get model usage from Redis
        top_models = []
        for model_name in llm_service.available_models.keys():
            count_key = f"localllm:model_usage:{model_name}"
            count = redis_client.get(count_key)
            if count:
                top_models.append({"model": model_name, "count": int(count)})

        # Sort by usage and get top 5
        top_models.sort(key=lambda x: x["count"], reverse=True)
        top_models = top_models[:5]

        # Get requests by endpoint
        requests_by_endpoint = {
            "/chat": int(chat_requests),
            "/reflect": int(reflect_requests),
            "/models": int(models_requests)
        }

        # Calculate average response time (from histogram)
        avg_response_time = 0.0
        try:
            # Get histogram sum and count
            histogram_sum = REQUEST_DURATION._sum._value
            histogram_count = REQUEST_DURATION._count._value
            if histogram_count > 0:
                avg_response_time = histogram_sum / histogram_count
        except:
            pass

        return {
            "total_requests": total_requests,
            "requests_24h": total_requests,  # TODO: Implement time-windowed counting
            "avg_response_time": round(avg_response_time, 2),
            "cache_hit_ratio": round(cache_hit_ratio, 3),
            "redis_hits": keyspace_hits,
            "redis_misses": keyspace_misses,
            "top_models": top_models,
            "requests_by_endpoint": requests_by_endpoint,
            "models_loaded": len(llm_service.available_models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        # Return minimal stats on error
        return {
            "total_requests": 0,
            "requests_24h": 0,
            "avg_response_time": 0.0,
            "cache_hit_ratio": 0.0,
            "redis_hits": 0,
            "redis_misses": 0,
            "top_models": [],
            "requests_by_endpoint": {},
            "models_loaded": len(llm_service.available_models),
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/admin/analyze-session/{conversation_id}")
async def analyze_session(conversation_id: str):
    """Analyze conversation session for improvement opportunities"""
    try:
        analysis = analyze_conversation_history(conversation_id)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing session {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/learning-insights")
async def get_learning_insights():
    """Get aggregated learning insights and improvement patterns"""
    try:
        insights = {
            "timestamp": datetime.now().isoformat(),
            "improvement_opportunities": {},
            "model_performance": {},
            "reflection_stats": {}
        }

        # Get improvement opportunity stats by type
        for opp_type in ["negative_feedback", "clarification", "repetition", "rephrasing"]:
            type_key = f"localllm:learning:stats:{opp_type}"
            stats = redis_client.hgetall(type_key)
            if stats:
                insights["improvement_opportunities"][opp_type] = {
                    "total_count": int(stats.get("count", 0)),
                    "severity_high": int(stats.get("severity_high", 0)),
                    "severity_medium": int(stats.get("severity_medium", 0)),
                    "severity_low": int(stats.get("severity_low", 0))
                }

        # Get model-specific performance stats
        models = ["qwen3:latest", "qwen3-coder:latest", "deepseek-r1:7b"]
        for model in models:
            model_stats_key = f"localllm:learning:model_stats:{model}"
            stats = redis_client.hgetall(model_stats_key)
            if stats:
                insights["model_performance"][model] = {
                    "total_issues": int(stats.get("total_issues", 0)),
                    "negative_feedback": int(stats.get("type_negative_feedback", 0)),
                    "clarifications": int(stats.get("type_clarification", 0)),
                    "repetitions": int(stats.get("type_repetition", 0)),
                    "rephrasing": int(stats.get("type_rephrasing", 0))
                }

            # Get reflection stats
            reflection_stats_key = f"localllm:reflection:stats:{model}"
            refl_stats = redis_client.hgetall(reflection_stats_key)
            if refl_stats:
                total = int(refl_stats.get("total_reflections", 0))
                low_conf = int(refl_stats.get("low_confidence_count", 0))
                insights["reflection_stats"][model] = {
                    "total_reflections": total,
                    "low_confidence_count": low_conf,
                    "improvement_rate": (low_conf / total * 100) if total > 0 else 0.0
                }

        # Calculate summary metrics
        total_opportunities = sum(
            opp["total_count"]
            for opp in insights["improvement_opportunities"].values()
        )

        insights["summary"] = {
            "total_improvement_opportunities": total_opportunities,
            "most_common_issue": max(
                insights["improvement_opportunities"].items(),
                key=lambda x: x[1]["total_count"],
                default=("none", {"total_count": 0})
            )[0] if total_opportunities > 0 else "none",
            "models_tracked": len([m for m, stats in insights["model_performance"].items() if stats["total_issues"] > 0])
        }

        return insights

    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/models/pull")
async def pull_model(model_name: str):
    """Pull a model from Ollama library"""
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        logger.info(f"Pulling model {model_name} from Ollama at {ollama_url}")

        # Call Ollama API to pull the model
        response = requests.post(
            f"{ollama_url}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600  # 10 minute timeout for model downloads
        )

        if response.status_code == 200:
            # Stream the download progress
            progress_data = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        progress_data.append(data)
                        if "status" in data:
                            logger.info(f"Pull progress: {data['status']}")
                    except json.JSONDecodeError:
                        pass

            logger.info(f"Successfully pulled model {model_name}")
            return {
                "status": "success",
                "message": f"Model {model_name} pulled successfully",
                "model": model_name,
                "progress": progress_data[-1] if progress_data else {}
            }
        else:
            logger.error(f"Failed to pull model {model_name}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Failed to pull model: {response.text}")

    except requests.exceptions.Timeout:
        logger.error(f"Timeout pulling model {model_name}")
        raise HTTPException(status_code=504, detail="Model pull timeout - download may still be in progress")
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model from Ollama"""
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        logger.info(f"Deleting model {model_name} from Ollama at {ollama_url}")

        # Call Ollama API to delete the model
        response = requests.delete(
            f"{ollama_url}/api/delete",
            json={"name": model_name},
            timeout=30
        )

        if response.status_code == 200:
            logger.info(f"Successfully deleted model {model_name}")
            # Refresh available models
            llm_service.load_models()
            return {
                "status": "success",
                "message": f"Model {model_name} deleted successfully",
                "model": model_name
            }
        else:
            logger.error(f"Failed to delete model {model_name}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Failed to delete model: {response.text}")

    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(llm_service.available_models)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()
    with REQUEST_DURATION.time():
        response = await llm_service.chat(request)
        # Track model usage in Redis
        try:
            model_key = f"localllm:model_usage:{request.model}"
            redis_client.incr(model_key)
        except Exception as e:
            logger.error(f"Failed to track model usage: {e}")

        # Store conversation history if conversation_id is provided
        if request.conversation_id:
            store_conversation_turn(
                conversation_id=request.conversation_id,
                user_message=request.message,
                ai_response=response.response,
                model=request.model
            )

        return response

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Stream chat responses using Server-Sent Events (SSE)"""
    REQUEST_COUNT.labels(method="POST", endpoint="/chat/stream").inc()

    # Track model usage
    try:
        model_key = f"localllm:model_usage:{request.model}"
        redis_client.incr(model_key)
    except Exception as e:
        logger.error(f"Failed to track model usage: {e}")

    return StreamingResponse(
        llm_service.chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/reflect", response_model=ReflectResponse)
async def reflect_endpoint(request: ReflectRequest, api_key: str = Depends(verify_api_key)):
    """Analyze and improve model responses using reflection"""
    REQUEST_COUNT.labels(method="POST", endpoint="/reflect").inc()

    # Build reflection prompt
    reflection_prompt = f"""Analyze this response for quality:

Original Query: {request.query}
Context: {request.context}
Model Response: {request.model_response}

Evaluate:
1. Accuracy: Is the information correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it easy to understand?
4. Helpfulness: Does it provide actionable information?

Respond in JSON format:
{{
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "helpfulness": 0.0-1.0,
    "issues": ["list of issues found"],
    "suggestions": ["list of improvements"],
    "improved_response": "better version if needed or empty string"
}}
"""

    try:
        # Call Ollama for analysis
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": request.model,
                "prompt": reflection_prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=120
        )

        if response.status_code == 200:
            analysis_text = response.json()["response"]

            # Try to parse JSON from response
            try:
                # Extract JSON if wrapped in markdown code blocks
                if "```json" in analysis_text:
                    analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis_text:
                    analysis_text = analysis_text.split("```")[1].split("```")[0].strip()

                analysis = json.loads(analysis_text)
                confidence = (
                    float(analysis.get("accuracy", 0.5)) +
                    float(analysis.get("completeness", 0.5)) +
                    float(analysis.get("clarity", 0.5)) +
                    float(analysis.get("helpfulness", 0.5))
                ) / 4.0

                return ReflectResponse(
                    analysis=analysis,
                    confidence=confidence,
                    suggestions=analysis.get("suggestions", []),
                    improved_response=analysis.get("improved_response", "")
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse reflection JSON: {e}")
                # Fallback if JSON parsing fails
                return ReflectResponse(
                    analysis={
                        "error": "Could not parse reflection",
                        "raw_response": analysis_text[:200]
                    },
                    confidence=0.5,
                    suggestions=["Review response manually"],
                    improved_response=""
                )
        else:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.status_code}")
    except Exception as e:
        logger.error(f"Reflection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reflection failed: {str(e)}")

@app.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    REQUEST_COUNT.labels(method="GET", endpoint="/models").inc()
    try:
        models_list = []
        for name, model in llm_service.available_models.items():
            # Debug: print what we're processing
            logger.info(f"Processing model: {name}, size: {model.get('size')}, type: {type(model.get('size'))}")
            
            # Ensure size is always a string
            size = model.get("size", "unknown")
            if isinstance(size, int):
                size = str(size)
                logger.info(f"Converted size from int to string: {size}")
            elif not isinstance(size, str):
                size = "unknown"
                logger.info(f"Size was not string, set to unknown: {size}")
            
            # Create ModelInfo with explicit string conversion
            model_info = ModelInfo(
                name=str(name),
                size=str(size),
                modified=str(model.get("modified_at", "unknown")),
                status="available"
            )
            models_list.append(model_info)
        
        return {"models": models_list}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        # Fallback: return basic model info without size
        return {
            "models": [
                ModelInfo(
                    name=str(name),
                    size="unknown",
                    modified="unknown",
                    status="available"
                )
                for name in llm_service.available_models.keys()
            ]
        }

@app.get("/metrics")
async def metrics():
    return generate_latest(REGISTRY)

# Background tasks
def schedule_model_updates():
    """Schedule background model update checks"""
    schedule.every(6).hours.do(llm_service.check_model_updates)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_background_tasks():
    """Start all background tasks"""
    # Start model update scheduler
    update_thread = threading.Thread(target=schedule_model_updates, daemon=True)
    update_thread.start()
    
    # Start periodic model loading
    load_thread = threading.Thread(target=periodic_model_loading, daemon=True)
    load_thread.start()

def periodic_model_loading():
    """Periodically reload available models"""
    while True:
        time.sleep(300)  # 5 minutes
        llm_service.load_models()

if __name__ == "__main__":
    # Start background tasks
    start_background_tasks()
    
    # Start the server
    uvicorn.run(
        "localllm.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        access_log=True
    )
