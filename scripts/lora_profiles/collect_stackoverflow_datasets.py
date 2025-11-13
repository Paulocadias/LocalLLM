"""
Stack Overflow Dataset Collection Script
Collects Q&A pairs from Stack Overflow for each coding profile

Usage:
    python collect_stackoverflow_datasets.py --profile bug_fixing --output datasets/lora_profiles/bug_fixing.jsonl --count 500
"""

import argparse
import json
import time
from typing import List, Dict
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stack Overflow API endpoint
SO_API_URL = "https://api.stackexchange.com/2.3/questions"

# Profile-specific tag configurations
PROFILE_TAGS = {
    "android": ["android", "kotlin", "java", "android-studio", "jetpack-compose"],
    "backend": ["python", "java", "node.js", "api", "rest", "sql", "database"],
    "frontend": ["javascript", "reactjs", "typescript", "html", "css", "react-hooks"],
    "bug_fixing": ["debugging", "exception", "error-handling", "troubleshooting"],
    "refactoring": ["code-review", "optimization", "refactoring", "best-practices"],
    "documentation": ["documentation", "comments", "code-explanation"]
}


def clean_html(html_text: str) -> str:
    """Remove HTML tags and clean text"""
    soup = BeautifulSoup(html_text, 'html.parser')

    # Extract code blocks
    code_blocks = []
    for code in soup.find_all('code'):
        code_blocks.append(code.get_text())
        code.replace_with(f"```{code.get_text()}```")

    text = soup.get_text()

    # Clean up whitespace
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

    return text


def fetch_stackoverflow_questions(tags: List[str], count: int = 100) -> List[Dict]:
    """Fetch questions from Stack Overflow API"""

    questions = []
    page = 1

    while len(questions) < count:
        params = {
            "page": page,
            "pagesize": 100,
            "order": "desc",
            "sort": "votes",
            "tagged": ";".join(tags[:3]),  # Use top 3 tags
            "site": "stackoverflow",
            "filter": "!9_bDDxJY5"  # Include body, answers, etc.
        }

        try:
            response = requests.get(SO_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if 'items' not in data or not data['items']:
                break

            questions.extend(data['items'])
            page += 1

            # Respect rate limits
            if 'backoff' in data:
                time.sleep(data['backoff'])
            else:
                time.sleep(1)

            logger.info(f"Fetched {len(questions)} questions so far...")

        except Exception as e:
            logger.error(f"Error fetching questions: {e}")
            break

    return questions[:count]


def get_question_answers(question_id: int) -> List[Dict]:
    """Get answers for a specific question"""

    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "!9_bDDxJY5"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        return data.get('items', [])

    except Exception as e:
        logger.error(f"Error fetching answers for {question_id}: {e}")
        return []


def process_question_answer_pair(question: Dict) -> Dict:
    """Process Q&A pair into instruction-response format"""

    # Get question text
    question_title = question.get('title', '')
    question_body = clean_html(question.get('body', ''))

    instruction = f"{question_title}\n\n{question_body}"

    # Get best answer
    answers = question.get('answers', [])
    if answers:
        # Sort by score and acceptance
        best_answer = max(answers, key=lambda a: (
            a.get('is_accepted', False),
            a.get('score', 0)
        ))

        response = clean_html(best_answer.get('body', ''))

        return {
            "instruction": instruction.strip(),
            "response": response.strip(),
            "score": question.get('score', 0),
            "answer_score": best_answer.get('score', 0),
            "is_accepted": best_answer.get('is_accepted', False)
        }

    return None


def collect_from_stackoverflow(profile: str, max_examples: int = 500) -> List[Dict]:
    """Collect Q&A pairs from Stack Overflow"""

    tags = PROFILE_TAGS[profile]
    logger.info(f"Fetching questions with tags: {tags}")

    questions = fetch_stackoverflow_questions(tags, max_examples * 2)
    logger.info(f"Found {len(questions)} questions")

    examples = []

    for question in questions:
        # Skip questions without accepted/high-scored answers
        if question.get('answer_count', 0) == 0:
            continue

        if question.get('score', 0) < 5:  # Minimum score threshold
            continue

        # Process Q&A pair
        pair = process_question_answer_pair(question)

        if pair and len(pair['response']) > 100:  # Minimum response length
            examples.append(pair)

            if len(examples) >= max_examples:
                break

        # Rate limiting
        time.sleep(0.5)

    logger.info(f"Collected {len(examples)} Q&A pairs")
    return examples


def save_dataset(examples: List[Dict], output_path: str):
    """Save examples to JSONL file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            # Remove metadata for training
            training_example = {
                "instruction": example["instruction"],
                "response": example["response"]
            }
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(examples)} examples to {output_path}")

    # Save metadata version
    metadata_path = output_file.with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Stack Overflow datasets")
    parser.add_argument("--profile", type=str, required=True,
                        choices=PROFILE_TAGS.keys(),
                        help="Coding profile to collect")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of examples to collect")

    args = parser.parse_args()

    logger.info(f"Collecting {args.count} examples for {args.profile} profile...")

    examples = collect_from_stackoverflow(
        profile=args.profile,
        max_examples=args.count
    )

    if examples:
        save_dataset(examples, args.output)
        logger.info(f"✓ Collection complete! {len(examples)} examples saved.")
    else:
        logger.error("No examples collected.")
