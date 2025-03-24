import logging
from typing import Dict, Iterable, List, Optional

import ollama
import streamlit as st

from src.constants import ASSYMETRIC_EMBEDDING, OLLAMA_MODEL_NAME
from src.embeddings import get_embedding_model
from src.opensearch import hybrid_search
from src.utils import setup_logging


#Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def ensure_model_pulled(model: str) -> bool:
    """
    Check if the model is already pulled and availabe localy.

    Args:
        model (str): Name of the model.

    Returns:
        bool: True if the model is already pulled and available localy, False otherwise.
    
    """
    try:
        available_models = ollama.list()
        if model in available_models:
            logger.info(f"Model {model} is already pulled and available localy.")
        else:
            logger.info(f"Model {model} is not available localy. pulling it now...")
            ollama.pull(model)
            logger.info(f"Model {model} has been pulled and now available localy.")
    
    except ollama.ResponseError as e:
        logger.error(f"Error while pulling model {model}: {e.error}")
        return False
    return True


def run_llama_streaming(prompt: str, temperature: float) -> Optional[Iterable[str]]:
    """
    
    Uses Ollama's Python library to run the LLaMA model with streaming enabled.

    Args:
        prompt (str): The prompt to send to the model.
        temperature (float): The response generation temperature.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
    
    """
    try:
        # Now attempt to stream the response from the model
        logger.info("Streaming response from LLaMA model.")
        stream = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature},
        )
    except ollama.ResponseError as e:
        logger.error(f"Error during streaming: {e.error}")
        return None

    return stream

def prompt_template(query: str, context: str, history:List[Dict[str, str]]) -> str:
    """
    Generates the prompt for the LLaMA model based on the query, context and history.

    Args:
        query (str): The user's query.
        context (str): The document context gathered from hybride search.
        history (List[Dict[Str, str]]): The user's previous conversation hystory.

    Returns:
        str: The prompt for the LLaMA model.
    """
    prompt = "You are a knowledgeable chatbot assistant. "
    if context:
        prompt += (
            "Use the following context to answer the question.\nContext:\n"
            + context
            + "\n"
        )
    else:
        prompt += "Answer questions to the best of your knowledge.\n"

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    logger.info("Prompt constructed with context and conversation history.")
    return prompt


def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable[str]]:
    """
    Generates a chatbot response by performing hybrid search and incorporating conversation history.

    Args:
        query (str): The user's query.
        use_hybrid_search (bool): Flag to enable hybrid search.
        num_results (int): The number of search results to include in context.
        temperature (float): The response generation temperature.
        Chat_history (Optional[List[Dict[str, str]]], optional): A generator yielding response chunks as strings. Defaults to None.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.

    """
    chat_history = chat_history or []
    max_history_messages = 10
    history = chat_history[-max_history_messages:]
    context = ""

    # Include hybrid search results if enabled
    if use_hybrid_search:
        logger.info("Performing hybrid search.")
        if ASSYMETRIC_EMBEDDING:
            prefixed_query = f"passage: {query}"
        else:
            prefixed_query = f"{query}"
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(
            prefixed_query
        ).tolist()  # Convert tensor to list of floats
        search_results = hybrid_search(query, query_embedding, top_k=num_results)
        logger.info("Hybrid search completed.")

        # Collect text from search results
        for i, result in enumerate(search_results):
            context += f"Document {i}:\n{result['_source']['text']}\n\n"

    # Generate prompt using the prompt_template function
    prompt = prompt_template(query, context, history)

    return run_llama_streaming(prompt, temperature)