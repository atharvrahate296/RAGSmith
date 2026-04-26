"""
RAGSmith – LLM inference service
Supports two providers, switchable via LLM_PROVIDER env var:

  ollama → Local Ollama (MIT)  — dev / self-hosted
  groq   → Groq Cloud API      — AWS deployment (free tier)

Uses `requests` for Groq calls — urllib gets blocked by Cloudflare (error 1010).
Ollama stays on urllib since it's local and has no Cloudflare.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Tuple

logger = logging.getLogger("ragsmith.llm")

SYSTEM_PROMPT = (
    "You are RAGSmith, a helpful AI assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say so honestly. "
    "Do not hallucinate or invent facts. Be concise and accurate.\n"
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def _requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise RuntimeError("requests not installed. Run: pip install requests") from exc


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    model: str = "",
    history: List[Tuple[str, str]] = None,  # List of (query, response)
) -> str:
    from config import get_settings

    cfg = get_settings()
    effective_model = model or cfg.effective_llm_model

    if cfg.llm_provider == "groq":
        return _groq_generate(query, context_chunks, effective_model, cfg.groq_api_key, history)

    return _ollama_generate(query, context_chunks, effective_model, cfg.ollama_base_url, history)


def check_llm_available() -> dict:
    from config import get_settings

    cfg = get_settings()

    if cfg.llm_provider == "groq":
        try:
            ok = _check_groq(cfg.groq_api_key)
            return {
                "available": ok,
                "provider": "groq",
                "detail": "Groq API reachable" if ok else "Groq API unreachable or invalid key",
            }
        except LLMError as exc:
            logger.error("Groq LLM check failed: %s", exc)
            return {"available": False, "provider": "groq", "detail": str(exc)}

    try:
        ok = _check_ollama(cfg.ollama_base_url)
        return {
            "available": ok,
            "provider": "ollama",
            "detail": f"Ollama at {cfg.ollama_base_url}" if ok else "Ollama not running",
        }
    except Exception as exc:
        logger.error("Ollama LLM check failed: %s", exc)
        return {"available": False, "provider": "ollama", "detail": str(exc)}


def list_available_models() -> List[str]:
    from config import get_settings

    cfg = get_settings()

    if cfg.llm_provider == "groq":
        return _groq_list_models(cfg.groq_api_key)

    return _ollama_list_models(cfg.ollama_base_url)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_messages(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    history: List[Tuple[str, str]] = None
) -> List[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    if context_chunks:
        parts = [
            f"[Source {i}: {fn} | relevance: {s:.3f}]\n{t}"
            for i, (t, s, fn) in enumerate(context_chunks, 1)
        ]
        ctx = "\n\n---\n\n".join(parts)
    else:
        ctx = "No relevant context found."

    user_content = f"CONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:"
    messages.append({"role": "user", "content": user_content})

    return messages


# ── Ollama ───────────────────────────────────────────────────────────────────

def _ollama_generate(query, context_chunks, model, base_url, history=None):
    payload = json.dumps({
        "model": model,
        "messages": _build_messages(query, context_chunks, history),
        "stream": True,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            parts = []
            for line in resp.read().decode().splitlines():
                if not line.strip():
                    continue
                obj = _try_json_parse(line)
                parts.append(obj.get("message", {}).get("content", ""))
                if obj.get("done"):
                    break

            response_text = "".join(parts).strip()

            if not response_text:
                logger.warning("Ollama returned empty response for model %s", model)
                return "Ollama returned an empty response."

            return response_text

    except urllib.error.URLError as exc:
        logger.error("Cannot reach Ollama at %s: %s", base_url, exc)
        raise LLMError(f"Cannot reach Ollama at {base_url}. Ensure it is running.") from exc

    except Exception as exc:
        logger.error("Unexpected Ollama error: %s", exc)
        raise LLMError(f"Ollama error: {exc}") from exc


def _check_ollama(base_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _ollama_list_models(base_url: str) -> List[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as r:
            data = _try_json_parse(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        raise LLMError(f"Ollama error: {exc}") from exc


# ── Groq ─────────────────────────────────────────────────────────────────────

def _groq_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class LLMError(Exception):
    pass


def _try_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON decode failed")
        return {}


def _groq_generate(query, context_chunks, model, api_key, history=None):
    if not api_key:
        raise LLMError("Groq API key is not set.")

    requests = _requests()

    payload = {
        "model": model,
        "messages": _build_messages(query, context_chunks, history),
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(api_key),
            json=payload,
            timeout=60,
        )

        resp.raise_for_status()
        result = resp.json()

        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"Groq HTTP error: {exc}") from exc

    except requests.exceptions.ConnectionError as exc:
        raise LLMError(f"Groq connection error: {exc}") from exc

    except requests.exceptions.Timeout as exc:
        raise LLMError("Groq timeout") from exc

    except Exception as exc:
        raise LLMError(f"Groq unexpected error: {exc}") from exc


def _check_groq(api_key: str) -> bool:
    if not api_key:
        raise LLMError("Groq API key not set")

    requests = _requests()
    resp = requests.get(GROQ_MODELS_URL, headers=_groq_headers(api_key), timeout=10)

    if resp.status_code == 200:
        return True

    if resp.status_code == 401:
        raise LLMError("Invalid Groq API key")

    return False


def _groq_list_models(api_key: str) -> List[str]:
    requests = _requests()

    resp = requests.get(GROQ_MODELS_URL, headers=_groq_headers(api_key), timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return [m["id"] for m in data.get("data", [])]