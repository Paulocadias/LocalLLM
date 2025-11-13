"""
Model Version Detector

Monitors for new model versions from:
1. Ollama library (via API)
2. HuggingFace model hub
3. Local Ollama installation

Automatically detects when new models become available.
"""

import requests
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersionDetector:
    """Detects new model versions from various sources"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.state_file = Path("model_versions_state.json")
        self.known_models = self.load_state()

    def load_state(self) -> Dict:
        """Load known model versions from state file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "last_check": None}

    def save_state(self):
        """Save current model state"""
        self.known_models["last_check"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.known_models, f, indent=2)

    def get_ollama_local_models(self) -> List[Dict]:
        """Get list of models from local Ollama installation"""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                logger.info(f"Found {len(models)} models in local Ollama")
                return models
            else:
                logger.error(f"Failed to get Ollama models: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []

    def get_ollama_library_models(self) -> List[Dict]:
        """
        Get available models from Ollama library

        Note: Ollama doesn't have a public API for browsing the library,
        so we check specific popular models.
        """

        # Popular models to check for updates
        popular_models = [
            "qwen2.5-coder:latest",
            "qwen2.5-coder:7b",
            "qwen2.5-coder:32b",
            "deepseek-coder-v2:latest",
            "deepseek-coder-v2:16b",
            "codellama:latest",
            "codellama:13b",
            "codellama:34b",
            "llama3:latest",
            "llama3:70b",
            "mistral:latest",
            "mixtral:latest",
            "phi3:latest"
        ]

        available_models = []

        for model_name in popular_models:
            try:
                # Check if model exists by attempting to get its info
                response = requests.post(
                    f"{self.ollama_url}/api/show",
                    json={"name": model_name},
                    timeout=5
                )

                if response.status_code == 200:
                    model_info = response.json()
                    available_models.append({
                        "name": model_name,
                        "available": True,
                        "info": model_info
                    })
                else:
                    # Model not available locally but might be in library
                    available_models.append({
                        "name": model_name,
                        "available": False,
                        "in_library": True
                    })

            except Exception as e:
                logger.debug(f"Error checking {model_name}: {e}")

        return available_models

    def detect_new_models(self) -> List[Dict]:
        """Detect new models that weren't known before"""

        current_models = self.get_ollama_local_models()
        new_models = []

        for model in current_models:
            model_name = model.get("name", "")
            model_digest = model.get("digest", "")

            # Check if this is a new model or new version
            if model_name not in self.known_models.get("models", {}):
                # Brand new model
                new_models.append({
                    "name": model_name,
                    "digest": model_digest,
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", ""),
                    "status": "new_model",
                    "detection_time": datetime.now().isoformat()
                })

                logger.info(f"Detected NEW MODEL: {model_name}")

            elif self.known_models["models"][model_name].get("digest") != model_digest:
                # Existing model with new version
                new_models.append({
                    "name": model_name,
                    "digest": model_digest,
                    "old_digest": self.known_models["models"][model_name].get("digest"),
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", ""),
                    "status": "updated_model",
                    "detection_time": datetime.now().isoformat()
                })

                logger.info(f"Detected MODEL UPDATE: {model_name}")

            # Update known models
            self.known_models.setdefault("models", {})[model_name] = {
                "digest": model_digest,
                "size": model.get("size", 0),
                "last_seen": datetime.now().isoformat()
            }

        # Save updated state
        self.save_state()

        return new_models

    def check_ollama_library_for_updates(self) -> List[Dict]:
        """
        Check Ollama library for models not yet downloaded
        Returns list of available models to download
        """

        library_models = self.get_ollama_library_models()
        local_models = self.get_ollama_local_models()
        local_model_names = {m.get("name", "") for m in local_models}

        available_downloads = []

        for model in library_models:
            model_name = model.get("name", "")

            # Check if model is in library but not local
            if model.get("in_library") and model_name not in local_model_names:
                available_downloads.append({
                    "name": model_name,
                    "status": "available_for_download",
                    "source": "ollama_library"
                })

        return available_downloads

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/show",
                json={"name": model_name},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a new model from Ollama library
        This is a long-running operation
        """
        try:
            logger.info(f"Pulling model: {model_name}")

            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minute timeout for large models
            )

            if response.status_code == 200:
                # Stream progress
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")

                        if "total" in data and "completed" in data:
                            total = data["total"]
                            completed = data["completed"]
                            percent = (completed / total * 100) if total > 0 else 0
                            logger.info(f"  {status}: {percent:.1f}%")
                        else:
                            logger.info(f"  {status}")

                logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def monitor_for_updates(self, check_interval_hours: int = 6):
        """
        Continuously monitor for model updates

        Args:
            check_interval_hours: How often to check (default: every 6 hours)
        """
        logger.info(f"Starting model version monitor (checking every {check_interval_hours} hours)")

        while True:
            try:
                logger.info("Checking for new model versions...")

                # Detect new/updated models
                new_models = self.detect_new_models()

                if new_models:
                    logger.info(f"Found {len(new_models)} new/updated models:")
                    for model in new_models:
                        logger.info(f"  - {model['name']} ({model['status']})")
                else:
                    logger.info("No new models detected")

                # Check library for available downloads
                available = self.check_ollama_library_for_updates()
                if available:
                    logger.info(f"Found {len(available)} models available for download:")
                    for model in available:
                        logger.info(f"  - {model['name']}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Wait before next check
            await asyncio.sleep(check_interval_hours * 3600)

    def get_status(self) -> Dict:
        """Get current detector status"""
        local_models = self.get_ollama_local_models()
        available_downloads = self.check_ollama_library_for_updates()

        return {
            "local_models_count": len(local_models),
            "available_downloads_count": len(available_downloads),
            "last_check": self.known_models.get("last_check"),
            "tracked_models": len(self.known_models.get("models", {})),
            "local_models": [m.get("name") for m in local_models],
            "available_downloads": [m.get("name") for m in available_downloads]
        }


def main():
    """Test the detector"""
    detector = ModelVersionDetector()

    print("="*80)
    print("MODEL VERSION DETECTOR - TEST")
    print("="*80)
    print()

    # Get current status
    status = detector.get_status()
    print(f"Local Models: {status['local_models_count']}")
    print(f"Available Downloads: {status['available_downloads_count']}")
    print(f"Last Check: {status['last_check']}")
    print()

    print("Local Models:")
    for model_name in status['local_models']:
        print(f"  - {model_name}")
    print()

    # Detect new models
    print("Checking for new/updated models...")
    new_models = detector.detect_new_models()

    if new_models:
        print(f"\nFound {len(new_models)} new/updated models:")
        for model in new_models:
            print(f"  - {model['name']} ({model['status']})")
    else:
        print("No new models detected")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
