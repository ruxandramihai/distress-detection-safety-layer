import os
import json
import time
import hashlib

from google import genai
from google.genai import types

_client = None

SAFETY_OFF = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in [
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]


def client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.data[r["key"]] = r
        self.f = open(path, "a")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, record):
        record["key"] = key
        self.data[key] = record
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()


def call_gemini(model, system, turns, temperature=0.0, schema=None, cache=None, repeat_id=0,
                safety_off=True, thinking_budget=0, max_retries=12):
    key = _key(model, system, turns, temperature, schema.__name__ if schema else None, repeat_id, safety_off)
    if cache is not None and cache.get(key):
        return cache.get(key)

    cfg = dict(system_instruction=system, temperature=temperature)
    if safety_off:
        cfg["safety_settings"] = SAFETY_OFF
    if schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = schema
    if thinking_budget is not None and "2.5" in model and "pro" not in model:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in turns]

    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            resp = client().models.generate_content(model=model, contents=contents,
                                                    config=types.GenerateContentConfig(**cfg))
            latency = (time.perf_counter() - start) * 1000
            break
        except Exception as e:
            wait = min(120, 2 ** attempt * 2)
            print(f"  Gemini error ({e.__class__.__name__}: {str(e)[:80]}), retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError("Gemini call failed after retries")

    text, blocked, reason = None, False, None
    try:
        text = resp.text
    except Exception:
        text = None
    if not text:
        blocked = True
        fb = getattr(resp, "prompt_feedback", None)
        reason = str(getattr(fb, "block_reason", None)) if fb else None
        if resp.candidates:
            reason = reason or str(resp.candidates[0].finish_reason)
    usage = getattr(resp, "usage_metadata", None)
    record = {"text": text, "latency_ms": latency, "blocked": blocked, "block_reason": reason,
              "input_tokens": getattr(usage, "prompt_token_count", None),
              "output_tokens": getattr(usage, "candidates_token_count", None)}
    if cache is not None:
        cache.put(key, record)
    return record
