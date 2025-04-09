import os

from openai import OpenAI

class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.openai_client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key or os.getenv("OPENAI_API_KEY") or "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")

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
    response, count = llm.request("hello", "meta-llama/llama-4-scout:free")
    print(response, count)
