import os
import time  # Importe o módulo time
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class MistralLLMClient:
    def __init__(self, api_key: str = None):
        """
        Inicializa o cliente Mistral (versão >= 1.0.0).

        Args:
            api_key (str, optional): Sua chave da API Mistral.
                                     Se None, tenta buscar da variável de ambiente 'MISTRAL_API_KEY' ou 'OPENAI_API_KEY'.
        """
        effective_api_key = api_key if api_key is not None else os.getenv('MISTRAL_API_KEY', os.getenv('OPENAI_API_KEY'))

        if effective_api_key is None:
            raise ValueError("Chave da API Mistral não fornecida nem encontrada nas variáveis de ambiente 'MISTRAL_API_KEY' ou 'OPENAI_API_KEY'.")

        self.mistral_client = Mistral(api_key=effective_api_key)

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, int]:
        """
        Envia uma requisição para a API da Mistral usando a nova estrutura.

        Args:
            payload (str): O prompt/conteúdo a ser enviado para o modelo.
            model (str): O nome do modelo Mistral a ser usado (ex: 'mistral-small-latest').
            temperature (float, optional): Controla a aleatoriedade da resposta. Defaults to 0.0.
            max_tokens (int, optional): O número máximo de tokens a serem gerados na resposta. Defaults to 1024.
    )  # Pausa de 2 segundos (ajuste conforme necessário)           
        Returns:
            tuple[str, int]: Uma tupla contendo a resposta do modelo e a contagem de tokens de completude.
        """
        messages = [{"role": "user", "content": payload}]

        try:
            completion = self.mistral_client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            final_response = completion.choices[0].message.content
            token_count = completion.usage.completion_tokens

            # <<< ADICIONAR A PAUSA AQUI >>>
            # Espera 1 segundo após a chamada bem-sucedida, antes de retornar
            time.sleep(2    )  # Pausa de 2 segundos (ajuste conforme necessário)           

            return final_response, token_count

        except Exception as e:
             # Se ocorrer um erro na API, também podemos adicionar uma pausa antes de relançar
             # Isso pode ajudar se o erro 429 ocorrer e o script tentar novamente (dependendo da lógica externa)
             print(f"Erro na API Mistral: {e}. Aguardando 1 segundo...")
             time.sleep(1)
             # Relança a exceção para que o script principal possa tratá-la
             raise e

if __name__ == "__main__":
    try:
        # Cria uma instância do cliente Mistral
        llm = MistralLLMClient()

        prompt = "Olá! Como você está hoje?"
        # Use um nome de modelo Mistral válido. Exemplos:
        # 'open-mistral-7b', 'open-mixtral-8x7b', 'mistral-small-latest',
        # 'mistral-medium-latest', 'mistral-large-latest', 'codestral-latest'
        modelo_mistral = "mistral-small-latest"

        print(f"Enviando prompt para o modelo: {modelo_mistral}...")
        response, count = llm.request(prompt, modelo_mistral)

        print("\n--- Resposta Recebida ---")
        print(response)
        print("------------------------")
        print(f"Tokens de Conclusão: {count}")

    except ValueError as ve:
        print(f"Erro de configuração: {ve}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        import traceback
        print("\n--- Traceback Detalhado ---")
        traceback.print_exc()
        print("--------------------------")