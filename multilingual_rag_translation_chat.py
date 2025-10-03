import os
from dotenv import load_dotenv
load_dotenv()

import time
import asyncio
import functools
from contextlib import contextmanager
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

# Enable LangSmith tracing via environment variables
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

translation_prompt_template = PromptTemplate(
    input_variables=["text", "target_language", "context"],
    template="""
You are a helpful assistant that translates text to a specified language.
If additional context is provided, use it to improve translation.

Context:
{context}

Translate this text into {target_language}:
{text}

Translation:
"""
)

class SimpleRetriever:
    def __init__(self):
        self.fake_context_db = {
            "weather": "Weather is related to atmospheric conditions like temperature, humidity, and precipitation.",
            "sports": "Sports include games like football, cricket, and basketball played competitively worldwide.",
            "technology": "Technology involves the use of scientific knowledge for practical purposes such as AI, computing, and engineering."
        }

    def retrieve(self, query):
        for keyword, context in self.fake_context_db.items():
            if keyword in query.lower():
                return context
        return "No relevant context found."

retriever = SimpleRetriever()

def trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[TRACE] Starting '{func.__name__}' with args={args} kwargs={kwargs}")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[TRACE] Finished '{func.__name__}' in {duration:.3f}s")
            return result
        except Exception as err:
            duration = time.time() - start
            print(f"[TRACE] Exception in '{func.__name__}' after {duration:.3f}s: {err}")
            raise
    return wrapper

@contextmanager
def trace_block(name):
    print(f"[TRACE] Entering block: {name}")
    start = time.time()
    try:
        yield
        duration = time.time() - start
        print(f"[TRACE] Exiting block: {name} (Elapsed: {duration:.3f}s)")
    except Exception as err:
        duration = time.time() - start
        print(f"[TRACE] Exception in block {name} after {duration:.3f}s: {err}")
        raise

@trace
def generate_prompt(text, target_language, context):
    prompt = translation_prompt_template.format(
        text=text, target_language=target_language, context=context
    )
    print(f"[TRACE] Generated prompt:\n{prompt}")
    return prompt

@trace
def retrieve_context(text):
    with trace_block("Retrieval"):
        context = retriever.retrieve(text)
    print(f"[TRACE] Retrieved context: {context}")
    return context

@trace
def translate_text(model, text, target_language):
    context = retrieve_context(text)
    prompt_text = generate_prompt(text, target_language, context)
    messages = [
        SystemMessage(content="You are a translation assistant with context support."),
        HumanMessage(content=prompt_text)
    ]
    with trace_block("Model Invocation"):
        response = model.invoke(messages)
    print(f"[TRACE] Model response: {response.content}")
    return response.content

def main():
    print("Welcome to the Continuous Interactive Multilingual Translation Chatbot!")
    print("Type 'exit' to quit, 'switch' to toggle between Groq and OpenAI models.\n")

    current_model = model_groq
    while True:
        user_text = input("Enter text to translate: ").strip()
        if user_text.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        if user_text.lower() == "switch":
            current_model = model_openai if current_model == model_groq else model_groq
            print(f"Switched to {'OpenAI GPT-4o' if current_model == model_openai else 'Groq GPT-oss-20b'} model.")
            continue

        target_language = input("Enter target language (e.g., French, Spanish, Hindi): ").strip()
        if not target_language:
            print("[ERROR] Target language cannot be empty.")
            continue

        with trace_block("Full Translation Request"):
            translation_result = translate_text(current_model, user_text, target_language)

        print(f"\nTranslation ({'OpenAI' if current_model == model_openai else 'Groq'}): {translation_result}\n")

if __name__ == "__main__":
    main()
