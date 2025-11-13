"""
Cline Prompt Parser - Extract user queries from Cline's verbose internal prompts

Cline sends massive prompts with internal instructions, environment details, etc.
This module extracts the actual user query so we can apply our enhancements
(reflection, caching, learning) without showing internal prompts.
"""

import re
import logging

logger = logging.getLogger(__name__)


class ClinePromptParser:
    """Parser for Cline's structured prompts"""

    def __init__(self):
        # Patterns for Cline's internal XML-like tags
        self.internal_tags = [
            'task', 'environment_details', 'system', 'tools', 'rules',
            'capabilities', 'ide_selection', 'commentary', 'example'
        ]

    def is_cline_prompt(self, message: str) -> bool:
        """
        Detect if a message is from Cline (contains internal tags)

        Args:
            message: The message to check

        Returns:
            True if this looks like a Cline prompt
        """
        if not message or len(message) < 100:
            return False

        # Check for common Cline markers
        cline_markers = [
            '<task>',
            '<environment_details>',
            'You are an AI assistant',
            'You are an interactive CLI tool',
            '<ide_selection>',
            'Available agent types'
        ]

        return any(marker in message for marker in cline_markers)

    def extract_user_query(self, message: str) -> str:
        """
        Extract the actual user query from Cline's verbose prompt

        Cline format typically:
        <task>USER QUERY HERE</task>
        <environment_details>... massive internal context...</environment_details>
        <system>... system prompts...</system>

        We want to extract just the USER QUERY part.

        Args:
            message: Full Cline prompt

        Returns:
            Extracted user query (clean)
        """
        if not self.is_cline_prompt(message):
            # Not a Cline prompt, return as-is
            return message

        logger.info("Detected Cline prompt, extracting user query [PRIVACY: details not logged]")

        # Strategy 1: Extract from <task> tag
        task_match = re.search(r'<task>(.*?)</task>', message, re.DOTALL)
        if task_match:
            query = task_match.group(1).strip()
            logger.info(f"Extracted query from <task> tag [length: {len(query)} chars]")
            return query

        # Strategy 2: Look for user message after system context
        # Cline often puts user message after all the internal stuff
        lines = message.split('\n')

        # Find where internal tags end
        last_tag_end = 0
        for i, line in enumerate(lines):
            if any(f'</{tag}>' in line for tag in self.internal_tags):
                last_tag_end = i

        # Get content after last tag
        if last_tag_end > 0 and last_tag_end < len(lines) - 1:
            potential_query = '\n'.join(lines[last_tag_end + 1:]).strip()

            # Remove any remaining XML tags
            potential_query = re.sub(r'<[^>]+>', '', potential_query).strip()

            if potential_query and len(potential_query) > 10:
                logger.info(f"Extracted query after tags [length: {len(potential_query)} chars]")
                return potential_query

        # Strategy 3: Extract first significant line that's not a tag
        for line in lines:
            line = line.strip()

            # Skip empty lines and lines with tags
            if not line or '<' in line:
                continue

            # Skip known system prompts
            skip_prefixes = [
                'You are',
                'Available agent types',
                'When using',
                'Usage notes',
                'IMPORTANT:',
                'Here is useful information'
            ]

            if any(line.startswith(prefix) for prefix in skip_prefixes):
                continue

            # This looks like user content
            if len(line) > 10:
                logger.info(f"Extracted first content line [length: {len(line)} chars]")
                return line

        # Fallback: return first 200 chars (avoid huge prompts)
        logger.warning("Could not extract clean query, using fallback")
        clean_message = re.sub(r'<[^>]+>.*?</[^>]+>', '', message, flags=re.DOTALL)
        clean_message = clean_message.strip()[:500]
        return clean_message if clean_message else message[:200]

    def should_apply_enhancements(self, extracted_query: str) -> bool:
        """
        Determine if we should apply our enhancements (reflection, caching, etc.)

        Some Cline queries might be too simple (like "continue") and don't need full processing

        Args:
            extracted_query: The extracted user query

        Returns:
            True if we should apply enhancements
        """
        # Skip enhancements for very short commands
        simple_commands = ['continue', 'yes', 'no', 'ok', 'done', 'next', 'stop']

        if extracted_query.lower().strip() in simple_commands:
            return False

        # Apply enhancements for everything else
        return len(extracted_query) > 10


def parse_cline_message(message: str) -> dict:
    """
    Convenience function to parse a Cline message

    Args:
        message: Raw message (might be from Cline)

    Returns:
        Dict with:
            - original: Original message
            - is_cline: Boolean if this is a Cline prompt
            - query: Extracted user query
            - apply_enhancements: Whether to apply reflection/caching/etc
    """
    parser = ClinePromptParser()

    is_cline = parser.is_cline_prompt(message)
    query = parser.extract_user_query(message) if is_cline else message
    apply_enhancements = parser.should_apply_enhancements(query)

    return {
        "original": message,
        "is_cline": is_cline,
        "query": query,
        "apply_enhancements": apply_enhancements,
        "original_length": len(message),
        "query_length": len(query)
    }
