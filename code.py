from openai import OpenAI
import os
from enum import Enum
from google import genai
import anthropic
import xai_sdk as xai
from mistralai import Mistral
import cohere
import transformers

TEMPERATURE = 0.7

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
grok_client = xai.Client(api_key=os.environ['XAI_API_KEY'])
mistral_client = Mistral(api_key='p55tshG7JimMjdrE7BoSF2t1msjlBl9S')
qwen_client = OpenAI(api_key="sk-45cd0de19cdd45d6a7c0ed06896637c7", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
cohere_client = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])

class LLM(Enum):
    CHATGPT = 1
    CLAUDE = 2
    GEMINI = 3
    GROK = 4
    MISTRAL = 5
    QWEN = 6
    COMMAND = 7
    GEMMA = 8

def ask(prompt, llm):
    """Send prompt to specified LLM and return response"""
    try:
        if llm == LLM.CHATGPT:
            response = openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-5-nano"
            )
            return response.choices[0].message.content
        elif llm == LLM.GEMINI:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt
            )
            return response.text
        elif llm == LLM.CLAUDE:
            message = anthropic_client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=1000,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        elif llm == LLM.GROK:
            chat = grok_client.chat.create(model="grok-3-mini")
            chat.append(xai.chat.user(prompt))
            response = chat.sample()
            return response.content
        elif llm == LLM.MISTRAL:
            response = mistral_client.chat.complete(
                model = 'mistral-medium-latest',
                temperature = TEMPERATURE,
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
            return response.choices[0].message.content
        elif llm == LLM.QWEN:
            completion = qwen_client.chat.completions.create(
                model="qwen-flash",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        elif llm == LLM.COMMAND:
            response = cohere_client.chat(
                model = 'command-r7b-12-2024',
                temperature = TEMPERATURE,
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.message.content[0].text
        elif llm == LLM.GEMMA:
            response = gemini_client.models.generate_content(
                model = "gemma-3-27b-it",
                contents = prompt
            )
            return response.text
    except Exception as e:
        print(f"Error with {llm.name}: {e}")
        raise e
    
if __name__ == "__main__":
    # code goes here
    print("Output done")