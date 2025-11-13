"""
Model Improvement Service

Automatic model improvement service that integrates with the orchestrator:
1. Monitors for new model versions
2. Automatically tests new models against commercial AI
3. Generates reflexion training data
4. Creates improved models
5. Reports improvement metrics to orchestrator
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
import asyncio
import redis
import json
from datetime import datetime

from model_version_detector import ModelVersionDetector
from automatic_tester import AutomaticTester

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Model Improvement Service",
    description="Automatic model detection, testing, and improvement",
    version="1.0.0"
)

# Configuration
CLAUDE_API_KEY = os.getenv("COMMERCIAL_AI_API_KEY")
if not CLAUDE_API_KEY:
    logger.warning("COMMERCIAL_AI_API_KEY not set - commercial AI comparison features will be disabled")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
META_ORCHESTRATOR_URL = os.getenv("META_ORCHESTRATOR_URL", "http://localhost:8004")
MODEL_MANAGER_URL = os.getenv("MODEL_MANAGER_URL", "http://localhost:8005")

# Initialize components
detector = ModelVersionDetector(ollama_url=OLLAMA_URL)
tester = AutomaticTester(claude_api_key=CLAUDE_API_KEY, ollama_url=OLLAMA_URL)

# Redis client for state management
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_available = True
except:
    redis_client = None
    redis_available = False
    logger.warning("Redis not available - state persistence disabled")

# Background task state
improvement_tasks = {}


# Pydantic models
class ModelTestRequest(BaseModel):
    model_name: str
    num_tests: int = 10
    create_improved: bool = True


class ModelPullRequest(BaseModel):
    model_name: str


class ImprovementStatus(BaseModel):
    task_id: str
    status: str
    model_name: str
    started_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict] = None


# === Health Check ===

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "model-improvement",
        "redis_available": redis_available,
        "detector_active": True,
        "tester_active": True
    }


# === Model Version Detection ===

@app.get("/models/detect-new")
async def detect_new_models():
    """
    Detect new or updated models in local Ollama installation

    Returns list of new/updated models since last check
    """
    try:
        new_models = detector.detect_new_models()

        # Store in Redis if available
        if redis_available:
            redis_client.set(
                "model_improvement:last_detection",
                json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "new_models": new_models
                }),
                ex=86400  # 24 hour expiration
            )

        return {
            "status": "success",
            "new_models_count": len(new_models),
            "new_models": new_models
        }

    except Exception as e:
        logger.error(f"Error detecting new models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_model_status():
    """Get current model detection status"""
    try:
        status = detector.get_status()
        return {
            "status": "success",
            "detector_status": status
        }
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available-downloads")
async def get_available_downloads():
    """Get models available for download from Ollama library"""
    try:
        available = detector.check_ollama_library_for_updates()
        return {
            "status": "success",
            "available_count": len(available),
            "available_models": available
        }
    except Exception as e:
        logger.error(f"Error checking available downloads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pull")
async def pull_model(request: ModelPullRequest, background_tasks: BackgroundTasks):
    """Pull a model from Ollama library"""
    try:
        # Run pull in background
        background_tasks.add_task(detector.pull_model, request.model_name)

        return {
            "status": "started",
            "message": f"Pulling model {request.model_name} in background",
            "model_name": request.model_name
        }
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Automatic Testing & Improvement ===

@app.post("/improve/test-model")
async def test_model(request: ModelTestRequest, background_tasks: BackgroundTasks):
    """
    Test a model and generate improvement training data

    Runs full comparison test against commercial AI and generates reflexion data
    """
    try:
        task_id = f"test_{request.model_name.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store task metadata
        improvement_tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "model_name": request.model_name,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "results": None
        }

        # Run improvement pipeline in background
        background_tasks.add_task(
            run_improvement_pipeline,
            task_id,
            request.model_name,
            request.num_tests,
            request.create_improved
        )

        return {
            "status": "started",
            "task_id": task_id,
            "message": f"Testing model {request.model_name}",
            "num_tests": request.num_tests
        }

    except Exception as e:
        logger.error(f"Error starting model test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/improve/status/{task_id}")
async def get_improvement_status(task_id: str):
    """Get status of an improvement task"""
    if task_id not in improvement_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return improvement_tasks[task_id]


@app.get("/improve/tasks")
async def list_improvement_tasks():
    """List all improvement tasks"""
    return {
        "status": "success",
        "tasks_count": len(improvement_tasks),
        "tasks": list(improvement_tasks.values())
    }


@app.post("/improve/automatic")
async def run_automatic_improvement(background_tasks: BackgroundTasks):
    """
    Run automatic improvement on all new models detected

    This is the main endpoint the orchestrator calls when new models are detected
    """
    try:
        # Detect new models
        new_models = detector.detect_new_models()

        if not new_models:
            return {
                "status": "no_new_models",
                "message": "No new models detected"
            }

        # Start improvement tasks for each new model
        task_ids = []

        for model in new_models:
            model_name = model["name"]
            task_id = f"auto_{model_name.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            improvement_tasks[task_id] = {
                "task_id": task_id,
                "status": "running",
                "model_name": model_name,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "results": None
            }

            background_tasks.add_task(
                run_improvement_pipeline,
                task_id,
                model_name,
                10,  # Full 10 tests
                True  # Create improved model
            )

            task_ids.append(task_id)

        return {
            "status": "started",
            "models_count": len(new_models),
            "task_ids": task_ids,
            "message": f"Started automatic improvement for {len(new_models)} models"
        }

    except Exception as e:
        logger.error(f"Error in automatic improvement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Integration with Orchestrator ===

@app.post("/orchestrator/report-improvement")
async def report_improvement_to_orchestrator(task_id: str):
    """
    Report improvement results to meta-orchestrator

    This updates the learning system with new model performance
    """
    if task_id not in improvement_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = improvement_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    try:
        # Report to meta-orchestrator
        import requests

        response = requests.post(
            f"{META_ORCHESTRATOR_URL}/performance/update",
            json={
                "model_name": task["model_name"],
                "improved_model_name": task["results"].get("improved_model_name"),
                "win_rate": task["results"]["statistics"]["win_rate"],
                "timestamp": task["completed_at"]
            },
            timeout=10
        )

        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Reported to orchestrator"
            }
        else:
            logger.warning(f"Failed to report to orchestrator: {response.status_code}")
            return {
                "status": "partial",
                "message": "Orchestrator unavailable"
            }

    except Exception as e:
        logger.error(f"Error reporting to orchestrator: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/orchestrator/model-recommendations")
async def get_model_recommendations():
    """
    Get model recommendations from model-manager

    This helps determine which models to prioritize for improvement
    """
    try:
        import requests

        response = requests.get(
            f"{MODEL_MANAGER_URL}/recommend-model",
            params={"task_type": "code_generation"},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=502, detail="Model manager unavailable")

    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Background Tasks ===

async def run_improvement_pipeline(
    task_id: str,
    model_name: str,
    num_tests: int,
    create_improved: bool
):
    """
    Background task to run improvement pipeline

    This is the main automatic improvement logic
    """
    try:
        logger.info(f"Starting improvement pipeline for {model_name} (task: {task_id})")

        # Run full pipeline
        results = tester.full_improvement_pipeline(
            model_name=model_name,
            num_tests=num_tests,
            create_improved=create_improved
        )

        # Update task status
        improvement_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": results
        })

        # Store in Redis if available
        if redis_available:
            redis_client.set(
                f"model_improvement:task:{task_id}",
                json.dumps(improvement_tasks[task_id]),
                ex=604800  # 7 day expiration
            )

        logger.info(f"Improvement pipeline completed for {model_name}")
        logger.info(f"Win Rate: {results['statistics']['win_rate']:.1f}%")

        # Auto-report to orchestrator
        try:
            await report_improvement_to_orchestrator(task_id)
        except:
            logger.warning("Failed to auto-report to orchestrator")

    except Exception as e:
        logger.error(f"Error in improvement pipeline: {e}")

        improvement_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })


# === Monitoring Loop ===

@app.on_event("startup")
async def startup_event():
    """Start background monitoring loop"""
    logger.info("Model Improvement Service starting...")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Redis available: {redis_available}")
    logger.info(f"Meta-orchestrator URL: {META_ORCHESTRATOR_URL}")

    # Start monitoring loop in background
    asyncio.create_task(monitoring_loop())


async def monitoring_loop():
    """
    Continuous monitoring loop

    Checks for new models every 6 hours and automatically starts improvement
    """
    check_interval_hours = 6

    logger.info(f"Starting monitoring loop (checking every {check_interval_hours} hours)")

    while True:
        try:
            await asyncio.sleep(check_interval_hours * 3600)

            logger.info("Monitoring: Checking for new models...")

            # Detect new models
            new_models = detector.detect_new_models()

            if new_models:
                logger.info(f"Monitoring: Found {len(new_models)} new models")

                # Automatically start improvement for each
                for model in new_models:
                    model_name = model["name"]
                    task_id = f"monitor_{model_name.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    logger.info(f"Monitoring: Starting automatic improvement for {model_name}")

                    improvement_tasks[task_id] = {
                        "task_id": task_id,
                        "status": "running",
                        "model_name": model_name,
                        "started_at": datetime.now().isoformat(),
                        "completed_at": None,
                        "results": None
                    }

                    # Run in background
                    asyncio.create_task(
                        run_improvement_pipeline(task_id, model_name, 10, True)
                    )

            else:
                logger.info("Monitoring: No new models detected")

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")


# === Main Entry Point ===

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8008))

    logger.info("="*80)
    logger.info("MODEL IMPROVEMENT SERVICE")
    logger.info("="*80)
    logger.info(f"Starting on port {port}")
    logger.info()

    uvicorn.run(app, host="0.0.0.0", port=port)
