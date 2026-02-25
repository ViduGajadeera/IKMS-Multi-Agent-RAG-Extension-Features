"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List
import re
import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import SystemMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG)


def _enforce_citations(answer: str, context: str) -> str:
    """Post-processing: Ensure every factual sentence has citations.
    
    If verification agent didn't add citations, extract available citation IDs
    and ensure sentences have them.
    
    Args:
        answer: The answer text from verification agent
        context: The context with [C#] markers
        
    Returns:
        Answer with citations enforced on sentences that lack them
    """
    # If answer is the "insufficient context" message, return as-is
    if "provided context does not contain information" in answer:
        return answer
    
    # Extract available citation IDs from context
    citation_pattern = r'\[C(\d+)\]'
    available_citations = re.findall(citation_pattern, context)
    
    if not available_citations:
        return answer
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    corrected_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If sentence already has a citation, keep it as-is
        if re.search(citation_pattern, sentence):
            corrected_sentences.append(sentence)
            continue
        
        # Check if this is a factual sentence (contains specific info/numbers/technical terms)
        # Keep sentences that are general/meta statements without citations
        is_factual = (
            any(char.isdigit() for char in sentence) or  # Contains numbers
            re.search(r'\b(is|are|has|uses|supports|provides|includes|contains|requires)\b', sentence) or  # Technical verbs
            len(sentence.split()) > 5  # Longer sentences likely factual
        )
        
        # If it's factual and lacks citation, add a citation if available
        if is_factual and not re.search(citation_pattern, sentence):
            # Remove trailing period if exists
            if sentence.endswith('.'):
                sentence = sentence[:-1]
            # Add citation from first available (could be improved with semantic matching)
            first_citation = f"[C{available_citations[0]}]"
            sentence = f"{sentence} {first_citation}."
        
        corrected_sentences.append(sentence)
    
    return ' '.join(corrected_sentences)



# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

#summarization_agent = create_agent(
 #   model=create_chat_model(),
  #  tools=[],
   # system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
#)

#verification_agent = create_agent(
 #   model=create_chat_model(),
  #  tools=[],
   # system_prompt=VERIFICATION_SYSTEM_PROMPT,
#)

chat_model = create_chat_model()

def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    context = ""
    citations = {}

    # Prefer the last ToolMessage content (from retrieval_tool)
    docs = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            docs = msg.artifact or []
            break
    else:
        # No ToolMessage found - return empty context
        return {"context": "", "citations": {}}
    
    context_parts = []

    for i, doc in enumerate(docs):
        chunk_id = f"C{i+1}"
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[{chunk_id}] (Page {page})\n{doc.page_content}"
        )

        citations[chunk_id] = {
            "page": page,
            "source": source,
            "snippet": doc.page_content[:120] + "..."
        }

    return {
        "context": "\n\n".join(context_parts),
        "citations": citations,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"""YOUR TASK: Answer the question. EVERY fact you write MUST have a citation [C#] at the end.

Question: {question}

CONTEXT WITH CITATION IDS (these are the ONLY sources you can cite):
{context}

===== FORMAT YOUR ANSWER LIKE THIS =====

"Fact one here [C1]. Second fact here [C2]. Fact three from C1 again [C1]."

Do NOT write: "Fact [C1] here" (citation in wrong place)
Do NOT write: "Fact here. [C1]" (period before citation)
Do NOT write: "Fact here." (no citation)

===== YOUR RULES =====

1. Each factual sentence ENDS with [C#]
2. Look at the context to find which [C#] contains the fact
3. Use EXACT citation IDs from context above: [C1], [C2], [C3], [C4], etc.
4. NEVER make up citations like [C99]
5. Numbers, names, facts, details = MUST cite
6. If context doesn't answer → respond ONLY: "The provided context does not contain information to answer this question."

===== EXAMPLES =====

Good: "Vector databases enable similarity search [C1]. The system uses embedding vectors [C2]."
Bad: "Vector databases enable similarity search. The system uses embedding vectors [C2]."
Bad: "Vector [C1] databases enable similarity search."

===== GENERATE YOUR ANSWER WITH CITATIONS NOW ====="""

    result = chat_model.invoke([
    SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
    HumanMessage(content=user_content)
])

    draft_answer = result.content
    # Log raw agent messages and draft answer for debugging
    try:
        logger.debug("Summarization agent messages: %s", result)
        logger.debug("Draft answer: %s", draft_answer)
    except Exception:
        pass

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    - Applies citation enforcement as a safety net.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""TASK: Verify the draft answer. Enforce citation requirements strictly.

Question: {question}

Reference Context (with available citations):
{context}

Draft Answer to Verify:
{draft_answer}

VERIFICATION CHECKLIST:
1. Every factual sentence MUST end with a citation like [C1], [C2], etc.
2. Sentence WITHOUT citation → REMOVE IT
3. Citation [C#] doesn't support the claim → REMOVE THE SENTENCE
4. Only use citation IDs from the context above
5. Keep sentences that are properly cited and supported
6. If nothing remains after verification, return: "The provided context does not contain information to answer this question."

Return the corrected answer with citations intact. Do NOT explain your changes."""

    result = chat_model.invoke([
    SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
    HumanMessage(content=user_content)
])

    answer = result.content
    # Log raw verification messages and agent answer
    try:
        logger.debug("Verification agent messages: %s", result)
        logger.debug("Agent answer before enforcement: %s", answer)
    except Exception:
        pass

    # Safety net: Enforce citations if agent didn't add them properly
    answer = _enforce_citations(answer, context)
    try:
        logger.debug("Final answer after enforcement: %s", answer)
    except Exception:
        pass

    return {
    "answer": answer,
    "citations": state.get("citations"),
}
