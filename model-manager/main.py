from fastapi import FastAPI, HTTPException
import httpx
import redis
import json
import asyncio
from typing import Dict, List, Optional
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

app = FastAPI(title="Model Manager Service")

# Scheduler for automatic upgrades
scheduler = AsyncIOScheduler()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis client
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Model configurations with enhanced descriptions and reflection capabilities
MODEL_CONFIGS = {
    "coder": {
        "current": "qwen3-coder:latest",
        "latest": "qwen3-coder:latest",
        "check_url": "https://ollama.com/library/qwen3-coder",
        "type": "coding",
        "description": "Specialized coding model optimized for programming tasks, code generation, debugging, and software development",
        "strengths": ["code generation", "debugging", "software architecture", "algorithm design"],
        "weaknesses": ["general knowledge", "creative writing"],
        "performance_metrics": {"coding_accuracy": 0.85, "reasoning": 0.78, "speed": 0.92},
        "reflection_enabled": True
    },
    "multimodal": {
        "current": "qwen2.5-vl:7b",
        "latest": "qwen2.5-vl:7b",
        "check_url": "https://ollama.com/library/qwen2.5-vl",
        "type": "multimodal",
        "description": "Vision-language model for image understanding, document analysis, and multimodal reasoning",
        "strengths": ["image analysis", "document processing", "visual reasoning", "table extraction"],
        "weaknesses": ["pure text tasks", "mathematical reasoning"],
        "performance_metrics": {"vision_accuracy": 0.82, "multimodal_reasoning": 0.79, "speed": 0.88},
        "reflection_enabled": True
    },
    "general": {
        "current": "qwen3:latest",
        "latest": "qwen3:latest",
        "check_url": "https://ollama.com/library/qwen3",
        "type": "general",
        "description": "General-purpose language model for chat, reasoning, and knowledge tasks",
        "strengths": ["conversation", "knowledge retrieval", "reasoning", "creative writing"],
        "weaknesses": ["specialized coding", "complex mathematics"],
        "performance_metrics": {"reasoning": 0.81, "knowledge": 0.85, "speed": 0.95},
        "reflection_enabled": True
    },
    "deepseek-r1": {
        "current": "deepseek-r1:7b",
        "latest": "deepseek-r1:7b",
        "check_url": "https://ollama.com/library/deepseek-r1",
        "type": "reasoning",
        "description": "Advanced reasoning model with reflection capabilities, optimized for complex problem-solving and step-by-step thinking",
        "strengths": ["complex reasoning", "mathematical problems", "logical thinking", "reflection"],
        "weaknesses": ["coding tasks", "multimodal tasks"],
        "performance_metrics": {"reasoning": 0.89, "mathematics": 0.87, "reflection_accuracy": 0.91},
        "reflection_enabled": True
    },
    "qwen3-coder": {
        "current": "qwen3-coder:latest",
        "latest": "qwen3-coder:latest",
        "check_url": "https://ollama.com/library/qwen3-coder",
        "type": "coding",
        "description": "Next-generation coding model with improved performance and enhanced code understanding",
        "strengths": ["advanced code generation", "debugging", "software patterns", "API integration"],
        "weaknesses": ["general knowledge", "creative tasks"],
        "performance_metrics": {"coding_accuracy": 0.88, "reasoning": 0.82, "speed": 0.94},
        "reflection_enabled": True,
        "upgrade_planned": "qwen3-coder:7b"
    },
    "qwen3-general": {
        "current": "qwen3:latest",
        "latest": "qwen3:latest",
        "check_url": "https://ollama.com/library/qwen3",
        "type": "general",
        "description": "Enhanced general-purpose model with improved reasoning and knowledge capabilities",
        "strengths": ["advanced reasoning", "knowledge synthesis", "creative tasks", "problem-solving"],
        "weaknesses": ["specialized domains", "mathematical proofs"],
        "performance_metrics": {"reasoning": 0.86, "knowledge": 0.89, "speed": 0.96},
        "reflection_enabled": True,
        "upgrade_planned": "qwen3:7b"
    }
}

async def scheduled_auto_upgrade():
    """Scheduled task to automatically check for and apply upgrades"""
    logger.info("Running scheduled auto-upgrade check")
    try:
        # Check for available upgrades
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8005/models/auto-upgrade")
            if response.status_code == 200:
                results = response.json()
                upgrades_applied = sum(1 for result in results["auto_upgrade_results"].values() if result.get("status") == "upgraded")
                if upgrades_applied > 0:
                    logger.info(f"Applied {upgrades_applied} automatic upgrades")
                else:
                    logger.info("No upgrades available in scheduled check")
            else:
                logger.warning("Failed to run scheduled auto-upgrade")
    except Exception as e:
        logger.error(f"Error in scheduled auto-upgrade: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model configurations in Redis and start scheduled tasks"""
    for model_type, config in MODEL_CONFIGS.items():
        # Convert lists to JSON strings for Redis storage
        redis_config = {}
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                redis_config[key] = json.dumps(value)
            else:
                redis_config[key] = str(value)
        redis_client.hset(f"model:{model_type}", mapping=redis_config)
    logger.info("Model configurations initialized")
    
    # Start scheduled tasks
    scheduler.add_job(
        scheduled_auto_upgrade,
        trigger=IntervalTrigger(hours=6),  # Check every 6 hours
        id="auto_upgrade_check",
        name="Auto-upgrade check every 6 hours"
    )
    scheduler.start()
    logger.info("Scheduled auto-upgrade task started (every 6 hours)")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "model-manager"}

@app.get("/models")
async def get_models():
    """Get all model configurations"""
    models = {}
    for model_type in MODEL_CONFIGS.keys():
        model_data = redis_client.hgetall(f"model:{model_type}")
        if model_data:
            models[model_type] = model_data
    return models

@app.get("/models/{model_type}")
async def get_model(model_type: str):
    """Get specific model configuration"""
    model_data = redis_client.hgetall(f"model:{model_type}")
    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model type {model_type} not found")
    return model_data

@app.post("/models/{model_type}/update")
async def update_model(model_type: str, new_model: str):
    """Update a model to a new version"""
    if model_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Model type {model_type} not found")
    
    # Check if model exists in Ollama
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://ollama:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_exists = any(model["name"] == new_model for model in models)
                
                if not model_exists:
                    # Download the new model
                    logger.info(f"Downloading new model: {new_model}")
                    pull_response = await client.post(
                        f"http://ollama:11434/api/pull",
                        json={"name": new_model}
                    )
                    
                    if pull_response.status_code != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download model {new_model}"
                        )
            
            # Update model configuration
            current_config = redis_client.hgetall(f"model:{model_type}")
            current_config["current"] = new_model
            current_config["latest"] = new_model
            
            redis_client.hset(f"model:{model_type}", mapping=current_config)
            
            logger.info(f"Updated {model_type} model to {new_model}")
            return {"message": f"Model {model_type} updated to {new_model}", "status": "success"}
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

@app.get("/models/{model_type}/check-updates")
async def check_model_updates(model_type: str):
    """Check for model updates (placeholder for actual version checking)"""
    if model_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Model type {model_type} not found")
    
    # In a real implementation, this would check Hugging Face or Ollama registry
    # For now, we'll return the current configuration
    model_data = redis_client.hgetall(f"model:{model_type}")
    
    # Simulate update check - in production, this would make API calls
    has_update = False  # Placeholder
    
    return {
        "current": model_data.get("current"),
        "latest_available": model_data.get("latest"),
        "has_update": has_update,
        "type": model_data.get("type")
    }

@app.post("/models/refresh-all")
async def refresh_all_models():
    """Refresh all model configurations and check for updates"""
    results = {}
    
    for model_type in MODEL_CONFIGS.keys():
        try:
            # Check for updates for each model type
            model_data = redis_client.hgetall(f"model:{model_type}")
            current_model = model_data.get("current")
            latest_model = model_data.get("latest")
            
            # Check if there's an upgrade planned and if it's available
            upgrade_planned = model_data.get("upgrade_planned")
            if upgrade_planned and upgrade_planned != current_model:
                # Check if the upgrade model is available in Ollama
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://ollama:11434/api/tags")
                    if response.status_code == 200:
                        available_models = response.json().get("models", [])
                        upgrade_available = any(model["name"] == upgrade_planned for model in available_models)
                        
                        if upgrade_available:
                            # Automatically upgrade to the new model
                            logger.info(f"Auto-upgrading {model_type} from {current_model} to {upgrade_planned}")
                            await update_model(model_type, upgrade_planned)
                            results[model_type] = {
                                "current": upgrade_planned,
                                "previous": current_model,
                                "status": "auto-upgraded",
                                "upgrade_type": "planned_upgrade"
                            }
                            continue
            
            # In production, this would check actual model registries
            # For now, we'll just return current status
            results[model_type] = {
                "current": current_model,
                "latest": latest_model,
                "status": "checked"
            }
            
        except Exception as e:
            results[model_type] = {
                "error": str(e),
                "status": "failed"
            }
    
    return {"refresh_results": results}

@app.post("/models/auto-upgrade")
async def auto_upgrade_models():
    """Automatically upgrade all models to their planned upgrades when available"""
    upgrade_results = {}
    
    for model_type, config in MODEL_CONFIGS.items():
        try:
            current_model = config["current"]
            upgrade_planned = config.get("upgrade_planned")
            
            if upgrade_planned and upgrade_planned != current_model:
                # Check if upgrade is available
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://ollama:11434/api/tags")
                    if response.status_code == 200:
                        available_models = response.json().get("models", [])
                        upgrade_available = any(model["name"] == upgrade_planned for model in available_models)
                        
                        if upgrade_available:
                            # Perform the upgrade
                            await update_model(model_type, upgrade_planned)
                            upgrade_results[model_type] = {
                                "from": current_model,
                                "to": upgrade_planned,
                                "status": "upgraded"
                            }
                        else:
                            upgrade_results[model_type] = {
                                "planned_upgrade": upgrade_planned,
                                "status": "not_available"
                            }
                    else:
                        upgrade_results[model_type] = {
                            "error": "Cannot check Ollama models",
                            "status": "failed"
                        }
            else:
                upgrade_results[model_type] = {
                    "status": "no_upgrade_planned"
                }
                
        except Exception as e:
            upgrade_results[model_type] = {
                "error": str(e),
                "status": "failed"
            }
    
    return {"auto_upgrade_results": upgrade_results}

@app.get("/recommend-model")
async def recommend_model(task_type: str, context: str = ""):
    """Recommend the best model for a given task type using enhanced descriptions and reflection"""
    # Enhanced model mapping with reflection capabilities
    model_mapping = {
        "coding": ["coder", "qwen3-coder"],
        "programming": ["coder", "qwen3-coder"],
        "multimodal": ["multimodal"],
        "vision": ["multimodal"],
        "general": ["general", "qwen3-general"],
        "chat": ["general", "qwen3-general"],
        "reasoning": ["deepseek-r1", "qwen3-general"],
        "mathematics": ["deepseek-r1"],
        "complex_problem": ["deepseek-r1"],
        "reflection": ["deepseek-r1"],
        "debugging": ["coder", "deepseek-r1"],
        "analysis": ["deepseek-r1", "general"]
    }
    
    # Get candidate models for this task type
    candidate_types = model_mapping.get(task_type.lower(), ["general", "qwen3-general"])
    
    # Score each candidate model based on task requirements
    scored_models = []
    for model_type in candidate_types:
        model_data = redis_client.hgetall(f"model:{model_type}")
        if model_data:
            score = calculate_model_score(model_data, task_type, context)
            scored_models.append({
                "model_type": model_type,
                "model_name": model_data.get("current"),
                "description": model_data.get("description"),
                "score": score,
                "strengths": model_data.get("strengths", []),
                "performance_metrics": model_data.get("performance_metrics", {})
            })
    
    # Sort by score and select the best
    scored_models.sort(key=lambda x: x["score"], reverse=True)
    best_model = scored_models[0] if scored_models else None
    
    if not best_model:
        # Fallback to general model
        model_data = redis_client.hgetall(f"model:general")
        best_model = {
            "model_type": "general",
            "model_name": model_data.get("current"),
            "description": model_data.get("description"),
            "score": 0.5,
            "reasoning": "Fallback to general model"
        }
    
    # Add reflection analysis
    reflection_analysis = generate_reflection_analysis(best_model, task_type, context, scored_models)
    
    return {
        "task_type": task_type,
        "recommended_model_type": best_model["model_type"],
        "recommended_model": best_model["model_name"],
        "score": best_model["score"],
        "description": best_model["description"],
        "reasoning": reflection_analysis,
        "alternative_models": scored_models[1:4] if len(scored_models) > 1 else [],
        "reflection_enabled": True
    }

def calculate_model_score(model_data: dict, task_type: str, context: str) -> float:
    """Calculate a score for how well a model matches the task requirements"""
    base_score = 0.5
    
    # Task type matching
    model_type = model_data.get("type", "")
    if task_type.lower() in model_type:
        base_score += 0.2
    
    # Strengths matching (handle JSON-encoded strings)
    strengths_str = model_data.get("strengths", "[]")
    try:
        strengths = json.loads(strengths_str)
    except:
        strengths = []
    
    for strength in strengths:
        if task_type.lower() in strength.lower():
            base_score += 0.1
    
    # Performance metrics (handle JSON-encoded strings)
    metrics_str = model_data.get("performance_metrics", "{}")
    try:
        metrics = json.loads(metrics_str)
    except:
        metrics = {}
    
    if "reasoning" in metrics and task_type in ["reasoning", "mathematics", "complex_problem"]:
        base_score += metrics["reasoning"] * 0.2
    if "coding_accuracy" in metrics and task_type in ["coding", "programming"]:
        base_score += metrics["coding_accuracy"] * 0.2
    if "vision_accuracy" in metrics and task_type in ["multimodal", "vision"]:
        base_score += metrics["vision_accuracy"] * 0.2
    
    # Context analysis
    if context:
        context_lower = context.lower()
        if "debug" in context_lower or "error" in context_lower:
            if "debugging" in strengths:
                base_score += 0.15
        if "complex" in context_lower or "difficult" in context_lower:
            if "deepseek-r1" in model_data.get("current", ""):
                base_score += 0.15
    
    # Reflection bonus
    reflection_enabled = model_data.get("reflection_enabled", "False")
    if reflection_enabled.lower() == "true":
        base_score += 0.1
    
    return min(base_score, 1.0)

def generate_reflection_analysis(best_model: dict, task_type: str, context: str, all_models: list) -> str:
    """Generate reflection analysis for model selection"""
    analysis = f"Selected {best_model['model_type']} model for {task_type} task with confidence score {best_model['score']:.2f}. "
    
    # Explain selection reasoning
    if best_model['score'] > 0.8:
        analysis += "High confidence selection based on strong task-model alignment. "
    elif best_model['score'] > 0.6:
        analysis += "Good confidence selection with appropriate task-model matching. "
    else:
        analysis += "Moderate confidence selection; consider alternatives if performance is suboptimal. "
    
    # Highlight key strengths
    strengths = best_model.get('strengths', [])
    if strengths:
        analysis += f"Key strengths for this task: {', '.join(strengths[:3])}. "
    
    # Mention alternatives if close scores
    if len(all_models) > 1:
        second_best = all_models[1]
        if best_model['score'] - second_best['score'] < 0.1:
            analysis += f"Close alternative: {second_best['model_type']} (score: {second_best['score']:.2f}). "
    
    # Reflection capabilities
    if best_model.get('model_type') == 'deepseek-r1':
        analysis += "This model includes advanced reflection capabilities for complex reasoning tasks. "
    
    return analysis

@app.get("/models/reflection/analyze-performance")
async def analyze_model_performance(model_type: str, task_results: str = ""):
    """Analyze model performance and provide reflection for improvement"""
    model_data = redis_client.hgetall(f"model:{model_type}")
    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model type {model_type} not found")
    
    reflection_analysis = {
        "model_type": model_type,
        "current_performance": model_data.get("performance_metrics", {}),
        "reflection_insights": [
            "Monitor response quality metrics over time",
            "Track task-specific success rates",
            "Consider model fine-tuning for specialized tasks",
            "Evaluate alternative models for edge cases"
        ],
        "improvement_suggestions": [
            "Gather more diverse training data for edge cases",
            "Implement A/B testing with alternative models",
            "Optimize prompt engineering for better results",
            "Consider ensemble approaches for critical tasks"
        ],
        "trust_metrics": {
            "confidence_score": 0.85,
            "reliability_rating": "high",
            "consistency_score": 0.82
        }
    }
    
    return reflection_analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
