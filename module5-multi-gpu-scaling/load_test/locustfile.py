import json
import os
import random

from locust import HttpUser, between, task


MODEL = os.environ.get("PLAYGROUND_LOADTEST_MODEL", "tinyllama-chat")
PROMPTS = [
    "Explain tensor parallelism in one paragraph.",
    "Write a short Python function that merges two sorted lists.",
    "Summarize the tradeoffs between pipeline and tensor parallelism for LLM inference.",
]


class ChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def chat_completions(self):
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": int(os.environ.get("PLAYGROUND_LOADTEST_MAX_TOKENS", "64")),
            "temperature": float(os.environ.get("PLAYGROUND_LOADTEST_TEMPERATURE", "0.7")),
        }
        with self.client.post(
            "/v1/chat/completions",
            data=json.dumps(body),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'fake-key')}",
            },
            catch_response=True,
            name="/v1/chat/completions",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:500]}")
            else:
                try:
                    data = resp.json()
                    _ = data["choices"][0]["message"]["content"]
                except (KeyError, ValueError, IndexError) as e:
                    resp.failure(f"bad json: {e}")

    @task(1)
    def completions_legacy(self):
        body = {
            "model": MODEL,
            "prompt": random.choice(PROMPTS),
            "max_tokens": int(os.environ.get("PLAYGROUND_LOADTEST_MAX_TOKENS", "64")),
        }
        with self.client.post(
            "/v1/completions",
            data=json.dumps(body),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'fake-key')}",
            },
            catch_response=True,
            name="/v1/completions",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:500]}")
