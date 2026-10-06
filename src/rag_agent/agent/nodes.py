from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


def query_rewrite_node(state: AgentState) -> dict:
    """Rewrite the user's question into a retrieval-friendly query."""

    messages = state.get("messages", [])

    original_query = ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            original_query = str(message.content)
            break

    if not original_query:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    try:
        settings = get_settings()
        llm = LLMFactory(settings).create()

        rewrite_prompt = (
            "Rewrite the following question into a short, keyword-rich "
            "search query for a deep learning study-material retrieval system. "
            "Keep the important technical terms. Return ONLY the rewritten "
            "search query, with no explanation.\n\n"
            f"Question: {original_query}"
        )

        response = llm.invoke(
            [HumanMessage(content=rewrite_prompt)]
        )

        rewritten = str(response.content).strip()

        if not rewritten:
            rewritten = original_query

    except Exception:
        rewritten = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten,
    }


def retrieval_node(state: AgentState) -> dict:
    """Retrieve relevant chunks from ChromaDB."""

    query = (
        state.get("rewritten_query")
        or state.get("original_query", "")
    )

    if not query:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    try:
        store = VectorStoreManager()

        chunks = store.query(
            query_text=query,
            topic_filter=state.get("topic_filter"),
            difficulty_filter=state.get("difficulty_filter"),
        )

    except Exception:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    if not chunks:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    return {
        "retrieved_chunks": chunks,
        "no_context_found": False,
    }


def generation_node(state: AgentState) -> dict:
    """Generate an answer using only the retrieved context."""

    settings = get_settings()
    llm = LLMFactory(settings).create()

    if state.get("no_context_found", False):
        answer = (
            "I was unable to find relevant information in the corpus "
            "for your query. Please try a more specific deep learning question."
        )

        response = AgentResponse(
            answer=answer,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )

        return {
            "final_response": response,
            "messages": [AIMessage(content=answer)],
        }

    retrieved_chunks = state.get("retrieved_chunks", [])

    context_parts = []
    sources = []
    scores = []

    for chunk in retrieved_chunks:
        citation = chunk.to_citation()

        context_parts.append(
            f"{citation}\n{chunk.chunk_text}"
        )

        if citation not in sources:
            sources.append(citation)

        scores.append(float(chunk.score))

    context = "\n\n".join(context_parts)

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "Use ONLY the retrieved study material below to answer "
                "the user's question. Do not invent information that is "
                "not supported by the retrieved context.\n\n"
                "Retrieved study material:\n"
                f"{context}"
            )
        ),
    ]

    prompt_messages.extend(
        state.get("messages", [])
    )

    prompt_messages.append(
        HumanMessage(
            content=state.get("original_query", "")
        )
    )

    result = llm.invoke(prompt_messages)

    answer = str(result.content).strip()

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", ""),
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)],
    }


def should_retry_retrieval(state: AgentState) -> str:
    """Decide whether retrieval succeeded."""

    if state.get("no_context_found", False):
        return "end"

    return "generate"
