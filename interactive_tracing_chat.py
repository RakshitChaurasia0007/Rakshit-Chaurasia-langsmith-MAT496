import os
from dotenv import load_dotenv
load_dotenv()  # Load any other env variables if present

import time
import asyncio
import functools
from contextlib import contextmanager
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage

# Enable LangSmith tracing via env vars
os.environ["LANGSMITHTRACING"] = "true"
os.environ["LANGSMITHPROJECT"] = "langsmith-academy"

# Initialize Groq and OpenAI chat models with API keys
model_groq = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    api_key="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"
)

model_openai = init_chat_model(
    "openai/gpt-4o",
    model_provider="openai",
    api_key="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"
)

print("Groq and OpenAI models initialized.")

# Tracing decorator (sync)
def trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[TRACE] Starting '{func.__name__}' with args {args} kwargs {kwargs}")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[TRACE] Finished '{func.__name__}' in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            print(f"[TRACE] Error in '{func.__name__}' after {duration:.3f}s: {e}")
            raise
    return wrapper

# Tracing context manager
@contextmanager
def trace_block(name):
    print(f"[TRACE] Enter block: {name}")
    start = time.time()
    try:
        yield
        duration = time.time() - start
        print(f"[TRACE] Exit block: {name} (Elapsed: {duration:.3f}s)")
    except Exception as e:
        duration = time.time() - start
        print(f"[TRACE] Exception in block {name} after {duration:.3f}s: {e}")
        raise

# Synchronous helper to invoke a model
@trace
def query_model(model, messages):
    return model.invoke(messages)

# Async tracing decorator
def async_trace(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"[TRACE] Async call '{func.__name__}' started")
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            print(f"[TRACE] Async call '{func.__name__}' finished in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            print(f"[TRACE] Async call '{func.__name__}' failed after {duration:.3f}s with error: {e}")
            raise
    return wrapper

@async_trace
async def async_chat_request(model, message_text, user="Tommy"):
    messages = [
        SystemMessage(content=f"User {user} session."),
        HumanMessage(content=message_text)
    ]
    # Run model invoke asynchronously using threads since invoke is blocking
    return await asyncio.to_thread(model.invoke, messages)

async def interactive_chat():
    print("\n=== Interactive Chat (type 'exit' to quit, 'switch' to toggle model) ===")
    current_model = model_groq
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        if user_input.strip().lower() == "switch":
            current_model = model_openai if current_model == model_groq else model_groq
            print(f"Switched to {'OpenAI GPT-4o' if current_model == model_openai else 'Groq GPT-oss-20b'} model.")
            continue
        response = await async_chat_request(current_model, user_input)
        print(f"AI ({'OpenAI' if current_model == model_openai else 'Groq'}): {response.content}")

if __name__ == "__main__":
    print("Starting traced interactive chat with Groq and OpenAI models.")
    print("Type 'switch' to change models, 'exit' to quit.\n")
    asyncio.run(interactive_chat())
