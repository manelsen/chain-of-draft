import argparse
import csv
import os
import traceback # Adicionado para melhor depuração

# Importa o cliente Mistral do seu módulo
from llm_mistral import MistralLLMClient
from tasks.coin_flip import CoinFlip
from tasks.date import DateUnderstanding
from tasks.gsm8k import GSM8K
from tasks.sports import SportsUnderstanding
from utils import average, nth_percentile

# Mapeamento de modelos atualizado para modelos Mistral
MODEL_MAPPING = {
    "mistral_small": "mistral-small-latest",
    "mistral_medium": "mistral-medium-latest",
    "mistral_large": "mistral-large-latest",
    "open_mistral": "open-mistral-7b",
    "open_mixtral": "open-mixtral-8x7b",
    "codestral": "codestral-latest",
    # Adicione ou remova modelos conforme necessário
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "date", "sports", "coin_flip"], required=True) # Tornar task obrigatório
    parser.add_argument("--model", default="mistral_small", choices=list(MODEL_MAPPING.keys())) # Usar chaves do mapping
    parser.add_argument(
        "--prompt",
        choices=["baseline", "cod", "cot"],
        default="cod",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Number of fewshot to be included, by default, include all fewshot examples",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base url for llm model endpoint (NÃO USADO pelo MistralLLMClient padrão)", # Nota adicionada
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model access, will use api keys in environment variables if not provided.",
    )

    args = parser.parse_args()

    # Instancia o cliente Mistral, passando apenas a api_key
    # O argumento args.url é ignorado aqui.
    try:
        llm_client = MistralLLMClient(api_key=args.api_key)
    except ValueError as ve:
        print(f"Erro ao inicializar o cliente LLM: {ve}")
        exit(1)
    except Exception as e:
        print(f"Erro inesperado ao inicializar o cliente LLM: {e}")
        traceback.print_exc()
        exit(1)


    match args.task:
        case "gsm8k":
            task = GSM8K(llm_client)
        case "date":
            task = DateUnderstanding(llm_client)
        case "sports":
            task = SportsUnderstanding(llm_client)
        case "coin_flip":
            task = CoinFlip(llm_client)
        case _:
            # Esta condição não deve ser atingida devido ao 'choices' e 'required=True' no parser
            print(f"Erro: Tarefa inválida '{args.task}'")
            exit(1)

    # Obtém o nome real do modelo Mistral a partir do mapeamento
    model_id = MODEL_MAPPING.get(args.model) # args.model agora é a chave (ex: "mistral_small")
    if not model_id:
         print(f"Erro: Chave de modelo inválida '{args.model}'. Modelos disponíveis: {list(MODEL_MAPPING.keys())}")
         exit(1)

    print(f"Avaliando tarefa '{args.task}' com o modelo '{model_id}' (chave: {args.model}) usando prompt '{args.prompt}'...")

    try:
        accuracy = task.evaluate(model_id, args.prompt, args.shot)

        # Verifica se houve execuções para calcular métricas
        if not task.latency_tracker:
             print("\nAvaliação concluída, mas nenhuma chamada de API foi registrada. Verifique a implementação da tarefa.")
             print(f"Acurácia: {accuracy}")
        else:
            results = [
                [
                    "Accuracy",
                    "Avg Token #",
                    "Average Latency (s)",
                    "P90 Latency (s)",
                    "P95 Latency (s)",
                    "P99 Latency (s)",
                ],
                [
                    accuracy,
                    average(task.token_count_tracker) if task.token_count_tracker else "N/A",
                    average(task.latency_tracker),
                    nth_percentile(task.latency_tracker, 0.9),
                    nth_percentile(task.latency_tracker, 0.95),
                    nth_percentile(task.latency_tracker, 0.99),
                ],
            ]
            print("\n--- Resultados ---")
            for i in range(len(results[0])):
                # Formata números float com 4 casas decimais
                value = results[1][i]
                if isinstance(value, (int, float)):
                     print(f"{results[0][i]}: {value:.4f}")
                else:
                     print(f"{results[0][i]}: {value}")
            print("------------------")


            if not os.path.exists("./results"):
                os.makedirs("./results")

            # Usa a chave do modelo (mais curta) para o nome do arquivo
            model_name_key = args.model # ex: "mistral_small"

            fname = (
                f"{args.task}-{model_name_key}-{args.prompt}-{args.shot}"
                if args.shot is not None # Verifica explicitamente se shot foi fornecido
                else f"{args.task}-{model_name_key}-{args.prompt}-allshots" # Nome mais claro se shot=None
            )
            filepath = f"./results/{fname}.csv"
            print(f"Salvando resultados em: {filepath}")
            with open(filepath, "w", newline='') as f: # Adicionado newline='' para evitar linhas em branco no CSV
                writer = csv.writer(f)
                writer.writerows(results)

    except Exception as e:
        print(f"\nErro durante a avaliação da tarefa '{args.task}': {e}")
        traceback.print_exc()
        exit(1)