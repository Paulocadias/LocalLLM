"""
GitHub Dataset Collection Script
Collects code examples from top GitHub repositories for each coding profile

Usage:
    python collect_github_datasets.py --profile android --output datasets/lora_profiles/android_mobile.jsonl --count 500
"""

import argparse
import json
import re
import time
from typing import List, Dict
from pathlib import Path
import requests
from github import Github
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub access token (set via environment or pass as argument)
# Get yours at: https://github.com/settings/tokens
GITHUB_TOKEN = None  # Set this or pass via --token argument

# Profile-specific search configurations
PROFILE_CONFIGS = {
    "android": {
        "languages": ["kotlin", "java"],
        "keywords": ["android", "recyclerview", "viewmodel", "jetpack compose"],
        "file_patterns": ["*Adapter.kt", "*ViewModel.kt", "*Activity.kt", "*Fragment.kt"],
        "stars_min": 500
    },
    "backend": {
        "languages": ["python", "java", "javascript", "go"],
        "keywords": ["api", "rest", "graphql", "database", "microservice"],
        "file_patterns": ["*controller*.py", "*service*.java", "*router*.js", "*handler*.go"],
        "stars_min": 1000
    },
    "frontend": {
        "languages": ["javascript", "typescript"],
        "keywords": ["react", "component", "hooks", "state"],
        "file_patterns": ["*Component.tsx", "*.jsx", "use*.ts"],
        "stars_min": 1000
    },
    "bug_fixing": {
        "languages": ["python", "java", "javascript"],
        "keywords": ["fix", "bug", "error", "exception"],
        "file_patterns": ["*test*.py", "*Test.java", "*.test.js"],
        "stars_min": 500
    },
    "refactoring": {
        "languages": ["python", "java", "javascript"],
        "keywords": ["refactor", "optimize", "improve"],
        "file_patterns": ["*.py", "*.java", "*.js"],
        "stars_min": 1000
    },
    "documentation": {
        "languages": ["python", "javascript", "java"],
        "keywords": ["readme", "docs", "guide", "tutorial"],
        "file_patterns": ["README.md", "CONTRIBUTING.md", "*.md"],
        "stars_min": 2000
    }
}


def clean_code(code: str) -> str:
    """Clean code by removing excessive whitespace"""
    # Remove multiple empty lines
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    # Trim lines
    code = '\n'.join(line.rstrip() for line in code.split('\n'))
    return code.strip()


def extract_function_or_class(content: str, language: str) -> List[str]:
    """Extract individual functions or classes from file content"""
    snippets = []

    if language in ["python"]:
        # Extract Python functions and classes
        pattern = r'((?:def|class)\s+\w+.*?(?=\n(?:def|class|\Z)))'
        matches = re.findall(pattern, content, re.DOTALL)
        snippets.extend(matches)

    elif language in ["kotlin", "java"]:
        # Extract Kotlin/Java classes and methods
        pattern = r'((?:class|fun|public|private|protected)\s+.*?\{.*?\n\})'
        matches = re.findall(pattern, content, re.DOTALL)
        snippets.extend(matches)

    elif language in ["javascript", "typescript"]:
        # Extract JS/TS functions and components
        pattern = r'((?:function|const|class)\s+\w+.*?(?=\n(?:function|const|class|export|\Z)))'
        matches = re.findall(pattern, content, re.DOTALL)
        snippets.extend(matches)

    return [clean_code(s) for s in snippets if len(s) > 50 and len(s) < 2000]


def generate_instruction(filename: str, code: str, profile: str) -> str:
    """Generate instruction prompt from filename and code"""

    # Extract meaningful name from filename
    name = Path(filename).stem

    if profile == "android":
        if "Adapter" in filename:
            return f"Create a RecyclerView adapter similar to {name}"
        elif "ViewModel" in filename:
            return f"Implement a ViewModel like {name}"
        elif "Activity" in filename:
            return f"Write an Activity class similar to {name}"

    elif profile == "backend":
        if "controller" in filename.lower():
            return f"Create a REST controller for {name}"
        elif "service" in filename.lower():
            return f"Implement a service class for {name}"
        elif "handler" in filename.lower():
            return f"Write an API handler for {name}"

    elif profile == "frontend":
        if "Component" in filename:
            return f"Create a React component {name}"
        elif "use" in filename:
            return f"Implement a custom hook {name}"

    # Generic fallback
    return f"Write code similar to this {name} implementation"


def collect_from_github(profile: str, max_examples: int = 500, token: str = None) -> List[Dict]:
    """Collect code examples from GitHub for the specified profile"""

    if not token:
        logger.error("GitHub token required. Get one at: https://github.com/settings/tokens")
        return []

    g = Github(token)
    config = PROFILE_CONFIGS[profile]
    examples = []

    for language in config["languages"]:
        logger.info(f"Searching {language} repositories...")

        # Build search query
        query = f"language:{language} stars:>{config['stars_min']}"
        for keyword in config["keywords"][:2]:  # Use top 2 keywords
            query += f" {keyword}"

        try:
            repos = g.search_repositories(query, sort="stars", order="desc")

            for repo in repos[:20]:  # Top 20 repos per language
                logger.info(f"Processing: {repo.full_name}")

                try:
                    # Get repository contents
                    contents = repo.get_contents("")

                    for file_pattern in config["file_patterns"]:
                        # Search for matching files
                        try:
                            results = repo.get_contents("", ref=repo.default_branch)

                            for content_file in results:
                                if content_file.type == "file" and any(
                                    content_file.name.endswith(ext)
                                    for ext in ['.py', '.java', '.kt', '.js', '.ts', '.tsx', '.jsx']
                                ):
                                    try:
                                        # Decode file content
                                        file_content = content_file.decoded_content.decode('utf-8')

                                        # Extract code snippets
                                        snippets = extract_function_or_class(file_content, language)

                                        for snippet in snippets:
                                            instruction = generate_instruction(
                                                content_file.name, snippet, profile
                                            )

                                            examples.append({
                                                "instruction": instruction,
                                                "response": snippet,
                                                "source": f"{repo.full_name}/{content_file.path}",
                                                "language": language
                                            })

                                            if len(examples) >= max_examples:
                                                return examples

                                    except Exception as e:
                                        logger.debug(f"Error processing file {content_file.name}: {e}")
                                        continue

                        except Exception as e:
                            logger.debug(f"Error searching pattern {file_pattern}: {e}")
                            continue

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching {language} repos: {e}")
            continue

    return examples


def save_dataset(examples: List[Dict], output_path: str):
    """Save examples to JSONL file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            # Remove source metadata for training
            training_example = {
                "instruction": example["instruction"],
                "response": example["response"]
            }
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(examples)} examples to {output_path}")

    # Also save full metadata version
    metadata_path = output_file.with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GitHub datasets for LoRA training")
    parser.add_argument("--profile", type=str, required=True,
                        choices=PROFILE_CONFIGS.keys(),
                        help="Coding profile to collect")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of examples to collect")
    parser.add_argument("--token", type=str,
                        help="GitHub access token")

    args = parser.parse_args()

    logger.info(f"Collecting {args.count} examples for {args.profile} profile...")

    examples = collect_from_github(
        profile=args.profile,
        max_examples=args.count,
        token=args.token
    )

    if examples:
        save_dataset(examples, args.output)
        logger.info(f"✓ Collection complete! {len(examples)} examples saved.")
    else:
        logger.error("No examples collected. Check GitHub token and try again.")
