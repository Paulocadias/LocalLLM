"""
Intelligent Model Routing

Advanced model selection for optimal coding quality
using DeepSeek-R1:14B (reasoning) + Qwen2.5-Coder:14B (coding) stack.

Optimized for AMD RX 6900 XT (16GB VRAM) with DirectML acceleration.
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Model Stack Configuration
REASONING_MODEL = "deepseek-reasoning"  # DeepSeek-R1:14B optimized
PRIMARY_CODER_MODEL = "qwen-coder"  # Qwen2.5-Coder:14B optimized
ADVANCED_CODER_MODEL = "qwen3-coder:30b"  # Optional for complex tasks
FALLBACK_MODEL = "qwen3:latest"  # Fallback if optimized models unavailable

class TaskComplexity:
    """Task complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelRouter:
    """
    Intelligent model router for production-grade setup.

    Routes tasks to the most appropriate model based on task type,
    complexity, and available VRAM/resources.
    """

    def __init__(self, enable_advanced_model: bool = False):
        """
        Initialize model router.

        Args:
            enable_advanced_model: Whether to use Qwen3-Coder:30B for complex tasks
                                   (requires 16GB+ VRAM with partial CPU offload)
        """
        self.enable_advanced_model = enable_advanced_model
        self.routing_stats = {
            "reasoning": 0,
            "coding": 0,
            "advanced": 0,
            "fallback": 0
        }

    def select_model(
        self,
        task_type: str,
        prompt: str,
        complexity: str = TaskComplexity.MEDIUM,
        context_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Select the best model for a given task.

        Args:
            task_type: Type of task (reasoning, coding, general, etc.)
            prompt: User prompt for additional context analysis
            complexity: Task complexity level
            context_length: Estimated context length needed

        Returns:
            Dictionary with model selection and configuration
        """

        # Analyze prompt for additional hints
        prompt_lower = prompt.lower()

        # === REASONING TASKS → DeepSeek-R1:14B ===
        reasoning_keywords = [
            "plan", "architect", "design", "review", "explain why",
            "analyze", "evaluate", "compare", "recommend", "strategy",
            "pros and cons", "best approach", "considerations",
            "trade-offs", "decision", "choose between"
        ]

        if task_type in ["reasoning", "planning", "architecture", "code_review", "debugging_strategy"]:
            self.routing_stats["reasoning"] += 1
            return {
                "model": REASONING_MODEL,
                "service": "localllm",
                "config": {
                    "temperature": 0.7,
                    "num_ctx": 16384 if context_length > 8192 else 8192,
                    "top_p": 0.9
                },
                "reason": "Reasoning task requires chain-of-thought and structured planning"
            }

        # Check for reasoning keywords in prompt
        if any(kw in prompt_lower for kw in reasoning_keywords):
            self.routing_stats["reasoning"] += 1
            return {
                "model": REASONING_MODEL,
                "service": "localllm",
                "config": {
                    "temperature": 0.7,
                    "num_ctx": 8192
                },
                "reason": "Detected reasoning/planning keywords in prompt"
            }

        # === CODE GENERATION TASKS ===
        coding_keywords = [
            "write", "create", "implement", "build", "generate",
            "function", "class", "method", "component", "module",
            "fix bug", "refactor", "optimize", "test"
        ]

        if task_type in ["code_generation", "refactoring", "bug_fixing", "testing"]:
            # High complexity coding → Advanced model (if enabled)
            if complexity == TaskComplexity.HIGH and self.enable_advanced_model:
                self.routing_stats["advanced"] += 1
                return {
                    "model": ADVANCED_CODER_MODEL,
                    "service": "langchain",
                    "config": {
                        "temperature": 0.3,
                        "num_ctx": 16384,
                        "num_gpu": 32  # Partial GPU offload for 30B model
                    },
                    "reason": "Complex coding task requires advanced 30B model"
                }

            # Standard coding → Primary coder
            self.routing_stats["coding"] += 1
            return {
                "model": PRIMARY_CODER_MODEL,
                "service": "langchain",
                "config": {
                    "temperature": 0.3,
                    "num_ctx": 8192,
                    "top_p": 0.9
                },
                "reason": "Code generation optimized with Qwen2.5-Coder"
            }

        # Check for coding keywords
        if any(kw in prompt_lower for kw in coding_keywords):
            self.routing_stats["coding"] += 1
            return {
                "model": PRIMARY_CODER_MODEL,
                "service": "langchain",
                "config": {
                    "temperature": 0.3,
                    "num_ctx": 8192
                },
                "reason": "Detected coding keywords in prompt"
            }

        # === MULTI-FILE REFACTORING → Advanced Model ===
        refactor_indicators = [
            "multi-file", "repository", "entire codebase", "refactor everything",
            "migrate", "restructure", "architectural change"
        ]

        if any(indicator in prompt_lower for indicator in refactor_indicators):
            if self.enable_advanced_model:
                self.routing_stats["advanced"] += 1
                return {
                    "model": ADVANCED_CODER_MODEL,
                    "service": "langchain",
                    "config": {
                        "temperature": 0.3,
                        "num_ctx": 16384,
                        "num_gpu": 32
                    },
                    "reason": "Large-scale refactoring requires advanced model with large context"
                }

        # === FALLBACK TO PRIMARY CODER ===
        # For general queries, explanations, etc.
        self.routing_stats["fallback"] += 1
        return {
            "model": PRIMARY_CODER_MODEL,
            "service": "localllm",
            "config": {
                "temperature": 0.5,
                "num_ctx": 8192
            },
            "reason": "Default to primary coder for general software engineering tasks"
        }

    def get_model_for_profile(self, profile: str) -> Optional[str]:
        """
        Get model for specific LoRA profile.

        Args:
            profile: LoRA profile name (android, backend, etc.)

        Returns:
            Model name or None if profile not found
        """

        # LoRA Profile Models (when trained)
        lora_models = {
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

        return lora_models.get(profile)

    def analyze_task_complexity(self, prompt: str, code_context: Optional[str] = None) -> str:
        """
        Analyze task complexity from prompt and context.

        Args:
            prompt: User prompt
            code_context: Optional code context

        Returns:
            Complexity level (low/medium/high)
        """

        high_complexity_indicators = [
            "multi-file", "entire codebase", "repository", "migrate",
            "architecture", "system design", "microservices",
            "distributed", "scalability", "performance critical"
        ]

        low_complexity_indicators = [
            "simple", "basic", "hello world", "quick", "small",
            "single function", "one file", "example"
        ]

        prompt_lower = prompt.lower()

        # Check high complexity
        if any(indicator in prompt_lower for indicator in high_complexity_indicators):
            return TaskComplexity.HIGH

        # Check low complexity
        if any(indicator in prompt_lower for indicator in low_complexity_indicators):
            return TaskComplexity.LOW

        # Check context length
        if code_context and len(code_context) > 5000:
            return TaskComplexity.HIGH

        # Default to medium
        return TaskComplexity.MEDIUM

    def get_routing_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return self.routing_stats.copy()

    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {key: 0 for key in self.routing_stats}


def create_model_router(enable_advanced_model: bool = False) -> ModelRouter:
    """
    Factory function to create model router.

    Args:
        enable_advanced_model: Whether to enable Qwen3-Coder:30B

    Returns:
        Configured ModelRouter instance
    """
    logger.info(f"Creating intelligent model router (advanced_model={enable_advanced_model})")
    return ModelRouter(enable_advanced_model=enable_advanced_model)


# Example usage
if __name__ == "__main__":
    router = create_model_router(enable_advanced_model=False)

    # Test reasoning task
    result = router.select_model(
        task_type="planning",
        prompt="Plan the architecture for a microservices-based e-commerce platform"
    )
    print("Reasoning task:", result)

    # Test coding task
    result = router.select_model(
        task_type="code_generation",
        prompt="Write a Python function to validate email addresses"
    )
    print("Coding task:", result)

    # Test complex refactoring
    result = router.select_model(
        task_type="code_generation",
        prompt="Refactor the entire codebase to use async/await patterns",
        complexity="high"
    )
    print("Complex refactor:", result)

    print("\nRouting stats:", router.get_routing_stats())
