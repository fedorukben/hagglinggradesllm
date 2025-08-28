from openai import OpenAI
import os
from enum import Enum
from google import genai
import anthropic
import xai_sdk as xai

TEMPERATURE = 0.7

# Initialize API clients with your original configurations
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
grok_client = xai.Client(api_key=os.environ['XAI_API_KEY'])
qwen_client = OpenAI(api_key="sk-45cd0de19cdd45d6a7c0ed06896637c7", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

class LLM(Enum):
    CHATGPT = 1
    CLAUDE = 2
    GEMINI = 3
    GROK = 4
    QWEN = 6

def ask(prompt, llm):
    """Send prompt to specified LLM and return response - using your original model names"""
    try:
        if llm == LLM.CHATGPT:
            response = openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-5-nano"  # Your original model name
            )
            return response.choices[0].message.content
        elif llm == LLM.GEMINI:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt  # Your original model name
            )
            return response.text
        elif llm == LLM.CLAUDE:
            message = anthropic_client.messages.create(
                model='claude-sonnet-4-20250514',  # Your original model name
                max_tokens=1000,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        elif llm == LLM.GROK:
            chat = grok_client.chat.create(model="grok-4")  # Your original model name
            chat.append(xai.chat.user(prompt))
            response = chat.sample()
            return response.content
        elif llm == LLM.QWEN:
            completion = qwen_client.chat.completions.create(
                model="qwen-flash",  # Your original model name
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
    except Exception as e:
        print(f"Error with {llm.name}: {e}")
        raise e
    
if __name__ == "__main__":
    # code goes here
    print("Output done")