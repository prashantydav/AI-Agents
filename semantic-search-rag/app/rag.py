import openai

from config import COMPLETION_MODEL, OPENAI_API_KEY


def _normalize_source(source: dict) -> str:
    title = source.get("metadata", {}).get("title", "source")
    return f"[{title}] {source.get('text', '').strip()}"


def _make_prompt(query: str, sources: list[dict]) -> str:
    source_text = "\n\n".join(_normalize_source(source) for source in sources)
    return f"""Use the following retrieved source snippets to answer the user's question.
Only answer from the provided sources and do not hallucinate.
If the answer is not contained in the sources, respond with: I don't know.

Source snippets:
{source_text}

Question: {query}

Answer concisely and cite sources in brackets using title or source id."""


def answer_query(query: str, sources: list[dict], model: str = COMPLETION_MODEL, temperature: float = 0.0) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to generate answers.")

    openai.api_key = OPENAI_API_KEY
    prompt = _make_prompt(query, sources)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a retrieval-augmented generation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response["choices"][0]["message"]["content"].strip()
