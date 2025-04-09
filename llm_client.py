import os

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        effective_base_url = base_url if base_url is not None else 'https://openrouter.ai/api/v1'
        self.openai_client = OpenAI(base_url=effective_base_url, api_key=os.getenv('OPENAI_API_KEY'))

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        ) -> tuple[str, int]:
        completion = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": payload}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        final_response = completion.choices[0].message.content
        token_count = completion.usage.completion_tokens
        return final_response, token_count

if __name__ == "__main__":
    llm = LLMClient()
    response, count = llm.request("hello", "google/gemini-2.0-flash-thinking-exp:free")
    print(response, count)
