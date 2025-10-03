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

# Enable tracing environment variables
os.environ["LANGSMITHTRACING"] = "true"
os.environ["LANGSMITHPROJECT"] = "langsmith-academy"

# Initialize models
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

# Prompt for stepwise math problem solving
math_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert math tutor. Solve the following problem step-by-step, explaining each step clearly:

Problem:
{question}

Stepwise Solution:
"""
)

# Tracing utilities
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
        except Exception as e:
            duration = time.time() - start
            print(f"[TRACE] Exception in '{func.__name__}' after {duration:.3f}s: {e}")
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
    except Exception as e:
        duration = time.time() - start
        print(f"[TRACE] Exception in block {name} after {duration:.3f}s: {e}")
        raise

@trace
def generate_prompt(question):
    prompt = math_prompt_template.format(question=question)
    print(f"[TRACE] Generated prompt:\n{prompt}")
    return prompt

@trace
def solve_math_problem(model, question):
    prompt_text = generate_prompt(question)
    messages = [
        SystemMessage(content="You are a helpful math tutor."),
        HumanMessage(content=prompt_text)
    ]
    with trace_block("Model Invocation"):
        response = model.invoke(messages)
    print(f"[TRACE] Model response:\n{response.content}")
    return response.content

def main():
    print("Interactive Stepwise Math Problem Solver")
    print("Type 'exit' to quit, 'switch' to toggle between Groq and OpenAI models.\n")

    current_model = model_groq

    while True:
        user_question = input("Enter a math problem: ").strip()
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        if user_question.lower() == "switch":
            current_model = model_openai if current_model == model_groq else model_groq
            print(f"Switched to {'OpenAI GPT-4o' if current_model == model_openai else 'Groq GPT-oss-20b'}")
            continue

        with trace_block("Full Math Problem Solving Request"):
            solution = solve_math_problem(current_model, user_question)

        print(f"\nSolution:\n{solution}\n")

if __name__ == "__main__":
    main()
