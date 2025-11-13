# Contributing to LocalLLM

Thank you for your interest in contributing to LocalLLM! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

This project adheres to a simple code of conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Git
- Basic knowledge of FastAPI, React, and Docker

### Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/localLLM.git
cd localLLM

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/localLLM.git
```

---

## Development Setup

### 1. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your local configuration
# (You don't need API keys for basic development)
```

### 2. Start Development Services
```bash
# Start all services
docker-compose -f docker-compose-simplified.yml up -d

# Or start specific services
docker-compose -f docker-compose-simplified.yml up localllm redis
```

### 3. Install Development Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-simplified.txt

# Install development tools
pip install pytest black flake8 mypy
```

---

## How to Contribute

### Types of Contributions

1. **Bug Reports**: Found a bug? Open an issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

2. **Feature Requests**: Have an idea? Open an issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered

3. **Code Contributions**: Ready to code?
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation updates

4. **Documentation**: Help others by improving:
   - README and guides
   - Code comments
   - API documentation
   - Example code

---

## Coding Standards

### Python Code Style
We follow PEP 8 with some modifications:

```python
# Use black for formatting
black localllm/ tests/

# Check style with flake8
flake8 localllm/ tests/ --max-line-length=100

# Type checking with mypy
mypy localllm/
```

### Code Quality Guidelines

1. **Clear and Descriptive Names**
```python
# Good
def calculate_win_rate(wins: int, total: int) -> float:
    return (wins / total) * 100

# Bad
def calc(w, t):
    return w/t*100
```

2. **Type Hints**
```python
# Always use type hints
def process_message(message: str, model: str = "qwen-coder") -> Dict[str, Any]:
    pass
```

3. **Docstrings**
```python
def train_lora_model(dataset_path: str, epochs: int = 3) -> str:
    """
    Train a LoRA model on the specified dataset.

    Args:
        dataset_path: Path to JSONL training dataset
        epochs: Number of training epochs (default: 3)

    Returns:
        Path to the trained model adapter

    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        ValueError: If epochs < 1
    """
    pass
```

4. **Error Handling**
```python
# Good: Specific exception handling
try:
    response = await call_model(message)
except ModelNotAvailableError:
    logger.warning(f"Model unavailable, falling back to default")
    response = await call_fallback_model(message)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### JavaScript/React Code Style

```javascript
// Use modern ES6+ syntax
const MyComponent = ({ message, onSend }) => {
  const [input, setInput] = useState('');

  const handleSubmit = useCallback(() => {
    onSend(input);
    setInput('');
  }, [input, onSend]);

  return (
    <div className="chat-input">
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={handleSubmit}>Send</button>
    </div>
  );
};
```

---

## Testing Guidelines

### Writing Tests

1. **Unit Tests**: Test individual functions
```python
# tests/test_orchestrator.py
import pytest
from meta_orchestrator.routing import select_profile

def test_select_profile_backend():
    """Test that backend keywords select backend profile"""
    message = "How do I optimize this database query?"
    profile = select_profile(message)
    assert profile == "qwen-backend"

def test_select_profile_frontend():
    """Test that frontend keywords select frontend profile"""
    message = "Create a React component with state"
    profile = select_profile(message)
    assert profile == "qwen-frontend"
```

2. **Integration Tests**: Test service interactions
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from localllm.main import app

client = TestClient(app)

def test_chat_endpoint():
    """Test /api/chat endpoint"""
    response = client.post(
        "/api/chat",
        json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_orchestrator.py

# Run with coverage
pytest --cov=localllm --cov-report=html
```

---

## Pull Request Process

### 1. Create a Branch
```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes
```bash
# Make your changes
# Test thoroughly
pytest

# Format code
black localllm/ tests/

# Check style
flake8 localllm/ tests/
```

### 3. Commit Changes
```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "Add: Feature description

- Detailed change 1
- Detailed change 2
- Fixes #123"
```

**Commit Message Format:**
```
<Type>: <Short description>

<Detailed description>
<GitHub issue reference if applicable>

Types: Add, Fix, Update, Remove, Refactor, Docs, Test
```

### 4. Push and Create PR
```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
# Fill out the PR template
```

### 5. PR Review Process
- Automated tests will run
- Maintainers will review your code
- Address feedback by pushing new commits
- Once approved, your PR will be merged

---

## Areas for Contribution

### High Priority

1. **Additional LoRA Profiles**
   - Data Science profile (pandas, numpy, visualization)
   - Security Auditing profile (vulnerability detection)
   - DevOps profile (Docker, Kubernetes, CI/CD)
   - Testing profile (pytest, test generation)

2. **Performance Optimizations**
   - Response caching improvements
   - Model loading optimization
   - Concurrent request handling

3. **Documentation**
   - Video tutorials
   - Architecture diagrams
   - API documentation
   - Deployment guides

### Medium Priority

4. **UI/UX Improvements**
   - Modern React dashboard
   - Real-time metrics
   - Conversation history search
   - Model switching interface

5. **Testing**
   - Increase test coverage
   - Integration tests
   - Performance benchmarks
   - Stress testing

6. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Error tracking (Sentry)
   - Usage analytics

### Nice to Have

7. **Model Support**
   - Llama 3 integration
   - Mistral integration
   - DeepSeek integration
   - Model comparison tool

8. **Advanced Features**
   - Multi-user authentication
   - RAG with document upload
   - Custom dataset builder UI
   - Fine-tuning from UI

---

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- Join our community chat (coming soon)

---

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project website (when available)

Thank you for contributing to LocalLLM! 🚀
