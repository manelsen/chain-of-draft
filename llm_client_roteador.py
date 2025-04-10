import os
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
# Precisa de:
# OPENAI_API_KEY=sk-or-v1... (sua chave OpenRouter)
# GOOGLE_API_KEY=SUA_CHAVE_GOOGLE_AI... (sua chave Google AI Studio)
load_dotenv()

# Define a URL padrão do OpenRouter
DEFAULT_OPENROUTER_URL = 'https://openrouter.ai/api/v1'

# Mapeamento dos modelos OpenRouter para modelos Google SDK específicos
# CORRIGIDO para usar os nomes exatos solicitados pelo usuário
GOOGLE_SDK_MODEL_MAP = {
    "google/gemini-2.0-flash-thinking-exp:free": "gemini-2.0-flash-thinking-exp-01-21",
    "google/gemini-2.0-flash-exp:free": "gemini-2.0-flash"
}

class LLMClient:
    """
    Cliente que atua como roteador: usa OpenRouter por padrão,
    mas direciona modelos específicos do Google para o SDK oficial do Google.
    """
    def __init__(self, base_url: str = None, openrouter_api_key: str = None, google_api_key: str = None):
        """
        Inicializa os clientes necessários.

        Args:
            base_url (str, optional): URL base para OpenRouter. Se None, usa DEFAULT_OPENROUTER_URL.
            openrouter_api_key (str, optional): Chave API para OpenRouter (compatível com OpenAI).
                                               Se None, busca de 'OPENAI_API_KEY' no .env.
            google_api_key (str, optional): Chave API para Google AI Studio.
                                            Se None, busca de 'GOOGLE_API_KEY' no .env.
        """
        # --- Configuração OpenRouter ---
        self.openrouter_base_url = base_url if base_url is not None else DEFAULT_OPENROUTER_URL
        self.openrouter_api_key = openrouter_api_key if openrouter_api_key is not None else os.getenv('OPENAI_API_KEY')

        if not self.openrouter_api_key:
            print("!!! AVISO: OPENAI_API_KEY (para OpenRouter) não encontrada. Chamadas para OpenRouter falharão.")
            self.openai_client = None # Não pode inicializar
        else:
            try:
                self.openai_client = OpenAI(
                    base_url=self.openrouter_base_url,
                    api_key=self.openrouter_api_key
                )
            except Exception as e:
                 print(f"!!! ERRO ao inicializar cliente OpenRouter: {e} !!!")
                 self.openai_client = None

        # --- Configuração Google SDK (apenas guarda a chave) ---
        self.google_api_key = google_api_key if google_api_key is not None else os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
             print("!!! AVISO: GOOGLE_API_KEY não encontrada. Chamadas diretas para Google SDK falharão.")
             self.google_key_available = False
        else:
             print(f"Chave Google AI encontrada. Key (fim): ...{self.google_api_key[-4:]}")
             self.google_key_available = True
             # Configura genai apenas quando for usar, no método _request_google

    def request(
        self,
        payload: str,
        model: str, # Nome do modelo como vindo do evaluate.py (formato OpenRouter)
        temperature: float = 0.7, # Default pode variar entre APIs
        max_tokens: int = 4096, # Nome genérico, adaptado para cada API
        safety_level: str = "BLOCK_MEDIUM_AND_ABOVE" # Específico do Google
        ) -> tuple[str, int]:
        """
        Envia a requisição para OpenRouter ou Google SDK baseado no nome do modelo.

        Args:
            payload (str): O prompt.
            model (str): Nome do modelo (formato OpenRouter).
            temperature (float): Temperatura.
            max_tokens (int): Máximo de tokens de saída.
            safety_level (str): Nível de segurança para Google SDK.

        Returns:
            tuple[str, int]: (resposta_texto, contagem_tokens)
                             Contagem de tokens é 0 para chamadas Google SDK.
        """

        # Verifica se o modelo solicitado deve ir para o Google SDK
        if model in GOOGLE_SDK_MODEL_MAP:
            google_sdk_model_name = GOOGLE_SDK_MODEL_MAP[model]
            # Adiciona o prefixo 'models/' se não estiver presente, comum no SDK
            if not google_sdk_model_name.startswith("models/"):
                google_sdk_model_name_prefixed = f"models/{google_sdk_model_name}"
            else:
                google_sdk_model_name_prefixed = google_sdk_model_name

            return self._request_google(
                payload=payload,
                model_name=google_sdk_model_name_prefixed, # Usa o nome prefixado
                temperature=temperature,
                max_output_tokens=max_tokens, # Adapta nome do parâmetro
                safety_level=safety_level
            )
        else:
            # Caso contrário, envia para OpenRouter
            return self._request_openrouter(
                payload=payload,
                model=model, # Usa o nome original para OpenRouter
                temperature=temperature,
                max_tokens=max_tokens
            )

    def _request_openrouter(
        self,
        payload: str,
        model: str,
        temperature: float,
        max_tokens: int
        ) -> tuple[str, int]:
        """Método privado para chamadas via OpenRouter."""
        if not self.openai_client:
            print("!!! ERRO: Cliente OpenRouter não inicializado. Verifique a chave OPENAI_API_KEY.")
            return "", 0

        final_response = ""
        token_count = 0
        try:
            completion = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # --- CORREÇÃO AQUI ---
            # Verifica se a resposta e a lista de 'choices' são válidas antes de acessar
            if completion and completion.choices:
                # Pega a primeira escolha
                choice = completion.choices[0]
                if choice.message and choice.message.content:
                    final_response = choice.message.content
                else:
                    print("!!! AVISO: Resposta OpenRouter com message.content vazio ou None.")

                if completion.usage:
                    token_count = getattr(completion.usage, 'completion_tokens', 0)

            else:
                print("!!! AVISO: Resposta OpenRouter recebida sem choices válidas.")

            return final_response, token_count

        except Exception as e:
            print("!!! ERRO na Requisição OpenRouter !!!")
            print(f"Erro ao fazer a requisição para o modelo {model} via OpenRouter: {e}")
            return "", 0

    def _request_google(
        self,
        payload: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        safety_level: str
        ) -> tuple[str, int]:
        """Método privado para chamadas via Google SDK."""
        if not self.google_key_available:
            print("!!! ERRO: Chave Google AI (GOOGLE_API_KEY) não disponível.")
            return "", 0

        try:
            # Configura o genai aqui, apenas quando necessário
            genai.configure(api_key=self.google_api_key)

            # Verifica se o nível de segurança é válido antes de usar
            try:
                block_threshold = HarmBlockThreshold[safety_level]
            except KeyError:
                block_threshold = HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: block_threshold,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: block_threshold,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: block_threshold,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: block_threshold,
            }
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            # Instancia o modelo usando o nome já prefixado
            model_instance = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            response = model_instance.generate_content(payload)

            # Tratamento de resposta bloqueada ou vazia
            if not response.candidates:
                print("!!! ATENÇÃO: Resposta Google SDK vazia e possivelmente bloqueada")
                try:
                    block_reason = response.prompt_feedback.block_reason
                    block_message = response.prompt_feedback.block_reason_message
                    print(f"Razão do Bloqueio: {block_reason} - {block_message}")
                except Exception:
                    try:
                         _ = response.text # Tenta acessar para ver se lança erro
                    except ValueError as ve:
                         print(f"Bloqueado por segurança (ValueError ao acessar .text): {ve}")
                    except AttributeError:
                         print("Não foi possível obter a razão do bloqueio (sem prompt_feedback).")
                    except Exception as fb_ex:
                         print(f"Erro inesperado ao tentar obter feedback de bloqueio: {fb_ex}")
                return "", 0

            final_response = response.text
            # print(f"--- Requisição Google AI OK ---")
            token_count = 0 # SDK não fornece contagem de conclusão facilmente
            return final_response, token_count

        except Exception as e:
            print(f"!!! ERRO na Requisição Google AI !!!")
            print(f"Erro ao fazer a requisição para o modelo {model_name} via Google SDK: {e}")
            if "API key not valid" in str(e):
                 print("Verifique se a GOOGLE_API_KEY está correta e ativa.")
            elif "permission" in str(e).lower() or "denied" in str(e).lower():
                 print(f"Verifique se sua chave Google tem permissão para usar o modelo '{model_name}'.")
            elif "404" in str(e) or "not found" in str(e).lower():
                 print(f"Modelo '{model_name}' não encontrado. Verifique o nome e a disponibilidade.")
            return "", 0

# Bloco para teste rápido
if __name__ == "__main__":
    print("\n--- Testando LLMClient Roteador diretamente (__main__) ---")
    try:
        client = LLMClient() # Inicializa buscando chaves do .env

        # --- Teste Rota OpenRouter ---
        print("\n--- Testando Rota OpenRouter ---")
        # Use um modelo que *não* está no GOOGLE_SDK_MODEL_MAP
        or_model = "meta-llama/llama-4-scout:free" # Exemplo
        or_prompt = "Explique OpenRouter em uma frase."
        if client.openai_client: # Verifica se o cliente OR foi inicializado
             or_response, or_tokens = client.request(or_prompt, or_model)
             print(f"\nResposta OpenRouter ({or_model}):\n{or_response}\nTokens: {or_tokens}")
        else:
             print("Skipping teste OpenRouter: cliente não inicializado.")

        # --- Teste Rota Google SDK ---
        # Use um modelo que *está* no GOOGLE_SDK_MODEL_MAP
        google_route_model_or_name = "google/gemini-2.0-flash-exp:free" # Este será roteado
        google_prompt = "Explique Google AI Studio em uma frase."
        if client.google_key_available: # Verifica se a chave Google está disponível
            # Obtem o nome mapeado para mostrar no log
            google_sdk_model_name_mapped = GOOGLE_SDK_MODEL_MAP.get(google_route_model_or_name, "N/A")
            google_response, google_tokens = client.request(google_prompt, google_route_model_or_name)
            print(f"\nResposta Google SDK ({google_route_model_or_name} -> {google_sdk_model_name_mapped}):\n{google_response}\nTokens: {google_tokens}")
        else:
            print("Skipping teste Google SDK: chave GOOGLE_API_KEY não disponível.")

    except ValueError as ve:
        print(f"Erro de configuração no teste __main__: {ve}")
    except Exception as ex:
        print(f"Ocorreu um erro inesperado durante o teste __main__: {ex}")