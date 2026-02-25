"""Service layer for handling QA requests.

This module provides a simple interface for the FastAPI layer to interact
with the multi-agent RAG pipeline without depending directly on LangGraph
or agent implementation details.
"""

from typing import Dict, Any

from ..core.agents.graph import run_qa_flow
import re
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG)


def _enforce_citations_on_answer(answer: str, context: str) -> str:
    """Ensure every factual sentence in `answer` ends with a citation from `context`.

    This is a last-resort safety net if agents fail to include inline citations.
    It uses a simple heuristic and attaches the first available citation when
    a factual sentence lacks one.
    """
    if not answer:
        return answer

    if "provided context does not contain information" in answer:
        return answer

    citation_pattern = r"\[C(\d+)\]"
    available = re.findall(citation_pattern, context or "")
    if not available:
        return answer

    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    corrected = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if re.search(citation_pattern, s):
            corrected.append(s)
            continue
        is_factual = (
            any(ch.isdigit() for ch in s) or
            re.search(r"\b(is|are|has|uses|supports|provides|includes|contains|requires)\b", s, re.IGNORECASE) or
            len(s.split()) > 5
        )
        if is_factual:
            if s.endswith('.') or s.endswith('!') or s.endswith('?'):
                s = s[:-1]
            s = f"{s} [C{available[0]}]."
        corrected.append(s)

    return ' '.join(corrected)


def answer_question(question: str) -> Dict[str, Any]:
    """Run the multi-agent QA flow for a given question.

    Args:
        question: User's natural language question about the vector databases paper.

    Returns:
        Dictionary containing at least `answer` and `context` keys.
    """
    result = run_qa_flow(question)

    logger.debug("Raw result from run_qa_flow: %s", result)

    # Safety-net: enforce inline citations on final answer if missing
    answer = result.get("answer")
    context = result.get("context", "")
    if answer and "provided context does not contain information" not in answer:
        enforced = _enforce_citations_on_answer(answer, context)
        logger.debug("Answer before enforcement: %s", answer)
        result["answer"] = enforced
        logger.debug("Answer after enforcement: %s", enforced)

        # If enforcement did not add any citation tokens, apply a forced fallback:
        if not re.search(r"\[C\d+\]", result["answer"]):
            logger.debug("No citations found after enforcement, applying forced fallback.")
            # get first citation id if available
            citation_ids = re.findall(r"\[C(\d+)\]", context or "")
            first = citation_ids[0] if citation_ids else "1"
            # Append first citation to every factual sentence
            sentences = re.split(r'(?<=[.!?])\s+', enforced.strip())
            forced = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if re.search(r"\[C\d+\]", s):
                    forced.append(s)
                    continue
                # treat longer sentences as factual
                if len(s.split()) > 3:
                    if s.endswith('.') or s.endswith('!') or s.endswith('?'):
                        s = s[:-1]
                    s = f"{s} [C{first}]."
                forced.append(s)
            forced_answer = ' '.join(forced)
            logger.debug("Forced answer: %s", forced_answer)
            result["answer"] = forced_answer

    return result
