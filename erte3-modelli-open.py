"""
================================================================================
SCRIPT BATCH COLAB DEEPINFRA: BENCHMARK e-RTE-3-it (3-CLASS)
================================================================================
Esegue la sequenza completa di esperimenti NLI su DeepInfra per i modelli:
- meta-llama/Llama-3.3-70B-Instruct-Turbo
- meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
- mistralai/Mistral-Small-24B-Instruct-2501

Protocolli testati:
- 0-shot
- 0-shot-alt
- N-shot-random-fixed (K in [3, 5, 8])
- N-shot-random-variable (K in [3, 5, 8])
- N-shot-semantic-minilm (K in [3, 5, 8])

Scrive i checkpoint FMT2 (JSON) ed i Log Raw (JSONL) nella cartella 'erte3_eval'.
================================================================================
"""

import json
import os
import time
from datetime import datetime
import requests
import pandas as pd
from google.colab import drive, userdata

# ==============================================================================
# 1. CONFIGURAZIONE ESPERIMENTI E AMBIENTE
# ==============================================================================
MODELS_TO_TEST = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemma-3-12b-it"
]

CHECKPOINT_EVERY = 64   # Salvataggio atomico FMT2
STATUS_EVERY = 64       # Messaggio di progresso a schermo
VERBOSE = False

TEMPERATURE = 0.0
MAX_TOKENS = 10

# Directory di lavoro dedicata a e-RTE-3-it
DRIVE_DIR = "/content/drive/MyDrive/erte3_eval"

# Dataset aumentato di input
DATASET_PATH = os.path.join(
    DRIVE_DIR,
    "e_rte3_it_BENCHMARK_FULL_3CLASS_UPTO8.csv"
)

# Modalita sperimentali
EVAL_MODES = ["0-shot", "0-shot-alt", "random-fixed", "random-variable", "semantic-minilm"]
K_SHOTS_TO_TEST = [3, 5, 8]

OVERWRITE = False

# ==============================================================================
# 2. INIZIALIZZAZIONE DEEPINFRA & GOOGLE DRIVE
# ==============================================================================
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

try:
    deepinfra_api_key = userdata.get("DEEPINFRA_API_KEY")
except Exception:
    deepinfra_api_key = os.environ.get("DEEPINFRA_API_KEY")

if not deepinfra_api_key:
    deepinfra_api_key = input("⚠️ Inserisci la tua DEEPINFRA_API_KEY: ")

DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {deepinfra_api_key}",
    "Content-Type": "application/json",
}

# ==============================================================================
# 3. HELPER FUNZIONI: PROMPT, SLICING & MAPPATURA NLI
# ==============================================================================
def slice_fewshot_context(full_context_str: str, k: int) -> str:
    """Estrae i primi K esempi dal contesto Few-Shot."""
    if not isinstance(full_context_str, str) or not full_context_str.strip():
        return ""
    blocks = full_context_str.strip().split("\n\n")
    return "\n\n".join(blocks[:k])

def clean_and_map_nli_prediction(raw_pred: str) -> str:
    """Mappa la risposta grezza nelle 3 classi standard."""
    clean = raw_pred.lower().strip()
    if "implicazione" in clean:
        return "Implicazione"
    elif "contraddizione" in clean:
        return "Contraddizione"
    elif "neutro" in clean:
        return "Neutro"
    else:
        return "UNKNOWN"

def build_nli_prompt(premise: str, hypothesis: str, mode: str, fewshot_ctx: str = "") -> str:
    """Costruisce la stringa di prompt esatta per il task NLI."""
    if mode == "0-shot":
        return (
            "Determina la relazione logica tra la Premessa e l'Ipotesi (Implicazione, Contraddizione, Neutro).\n"
            "Rispondi SOLTANTO con una parola.\n\n"
            f'Premessa: "{premise}"\n'
            f'Ipotesi: "{hypothesis}"\n'
            "Risposta:"
        )
    elif mode == "0-shot-alt":
        return (
            "Analizza la Premessa e l'Ipotesi seguenti e indica la relazione semantica scegliendo unicamente tra: Implicazione, Contraddizione, Neutro.\n"
            "Restituisci solo l'etichetta senza altri commenti.\n\n"
            f'Premessa: "{premise}"\n'
            f'Ipotesi: "{hypothesis}"\n'
            "Relazione:"
        )
    else:
        return (
            "Determina la relazione logica tra la Premessa e l'Ipotesi (Implicazione, Contraddizione, Neutro).\n"
            "Di seguito trovi alcuni esempi di riferimento per orientarti:\n\n"
            f"{fewshot_ctx}\n\n"
            "Ora analizza questa coppia:\n"
            f'Premessa: "{premise}"\n'
            f'Ipotesi: "{hypothesis}"\n'
            "Rispondi SOLTANTO con una parola (Implicazione, Contraddizione, Neutro).\n"
            "Risposta:"
        )

def predict_nli_deepinfra(prompt_text: str, model_id: str) -> tuple[str, float]:
    """Invia la richiesta al modello tramite l'API OpenAI-compatibile di DeepInfra."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            response = requests.post(
                url=DEEPINFRA_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=30
            )
            latency = time.time() - t0

            if response.status_code == 200:
                res_json = response.json()
                raw_response = res_json["choices"][0]["message"]["content"].strip()
                return raw_response, latency
            else:
                raise ValueError(f"HTTP Status {response.status_code}: {response.text}")

        except Exception as e:
            wait_time = 2.0 * (2 ** (attempt - 1))
            print(f"\n⚠️ [ERRORE DEEPINFRA] Attempt {attempt}/5. Attesa {wait_time}s... Errore: {e}")
            time.sleep(wait_time)

    return "ERRORE", time.time() - t0

# ==============================================================================
# 4. CARICAMENTO DATASET
# ==============================================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"❌ Impossibile trovare il dataset NLI in: {DATASET_PATH}. "
        "Esegui prima lo script di generazione locale e-RTE-3-it."
    )

df_test = pd.read_csv(DATASET_PATH)
total_records = len(df_test)
pad_width = len(str(total_records))

print(f"📊 Dataset e-RTE-3-it Caricato: {total_records} coppie da {DATASET_PATH}\n")

# ==============================================================================
# 5. CICLO MASTER ESPERIMENTI MULTI-MODELLO SU DEEPINFRA
# ==============================================================================
experiments_list = []
for mode in EVAL_MODES:
    if mode in ["0-shot", "0-shot-alt"]:
        experiments_list.append((mode, 0, mode))
    else:
        for k in K_SHOTS_TO_TEST:
            protocol_name = f"{k}-shot-{mode}"
            experiments_list.append((mode, k, protocol_name))

total_pipeline_runs = len(MODELS_TO_TEST) * len(experiments_list)
print(f"🚀 AVVIO BENCHMARK e-RTE-3-it SU DEEPINFRA ({len(MODELS_TO_TEST)} MODELLI x {len(experiments_list)} PROTOCOLLI = {total_pipeline_runs} CONFIGURAZIONI TOTALI)\n")

global_exp_idx = 1
for api_model in MODELS_TO_TEST:
    model_slug = api_model.replace('/', '_').replace('-', '_').replace('.', '_')

    print(f"\n" + "#" * 80)
    print(f"🤖 MODELLO IN ELABORAZIONE: {api_model}")
    print("#" * 80 + "\n")

    for eval_mode, k_shots, protocol_mode in experiments_list:
        checkpoint_file = os.path.join(
            DRIVE_DIR, f"checkpoint-fmt2-{protocol_mode}-erte3-{model_slug}.json"
        )
        raw_log_file = os.path.join(
            DRIVE_DIR, f"log-raw-{protocol_mode}-erte3-{model_slug}.jsonl"
        )

        print("=" * 80)
        print(f"▶️ TASK {global_exp_idx}/{total_pipeline_runs} | MODELLO: {api_model} | PROTOCOLLO: [{protocol_mode.upper()}]")
        print(f"📁 Checkpoint FMT2: {checkpoint_file}")
        print(f"📝 Raw Log File:    {raw_log_file}")
        print("=" * 80)

        if OVERWRITE:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            if os.path.exists(raw_log_file):
                os.remove(raw_log_file)

        results = []
        processed_ids = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    results = existing_data.get("results", [])
                    processed_ids = {str(r["id"]) for r in results}
                print(f"🔄 RIPRESA AUTOMATICA: {len(processed_ids)}/{total_records} coppie già elaborate.")
            except Exception as e:
                print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero.")

        if len(results) == total_records:
            print(f"✅ Configurazione [{protocol_mode.upper()}] per [{api_model}] già COMPLETATA al 100%. Salto.\n")
            global_exp_idx += 1
            continue

        fmt2_data = {
            "metadata": {
                "model": {
                    "provider": "deepinfra",
                    "model_id": api_model
                },
                "protocol": {
                    "task": "NLI",
                    "mode": protocol_mode
                },
                "experiment": {
                    "eval_mode": eval_mode,
                    "k_shots": k_shots,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "status_every": STATUS_EVERY
                },
                "dataset": {
                    "path": DATASET_PATH,
                    "total_records": total_records
                }
            },
            "results": results,
        }

        # Selezione dinamica colonna contesto
        fewshot_col = ""
        if eval_mode == "random-fixed":
            fewshot_col = "fewshot_random_3class_context"
        elif eval_mode == "random-variable":
            fewshot_col = "fewshot_random_variable_3class_context"
        elif eval_mode == "semantic-minilm":
            fewshot_col = "fewshot_semantic_minilm_3class_context"

        start_time = time.time()

        try:
            for idx, row in df_test.iterrows():
                rec_id = str(row["id"])
                if rec_id in processed_ids:
                    continue

                premise_text = str(row["premise"])
                hypothesis_text = str(row["hypothesis"])
                target_label = str(row["label_gold"])

                k_shot_ctx = ""
                if eval_mode not in ["0-shot", "0-shot-alt"] and fewshot_col in row:
                    full_ctx = str(row[fewshot_col])
                    k_shot_ctx = slice_fewshot_context(full_ctx, k_shots)

                prompt_raw_text = build_nli_prompt(premise_text, hypothesis_text, eval_mode, k_shot_ctx)
                raw_pred, latency = predict_nli_deepinfra(prompt_raw_text, api_model)

                if raw_pred == "ERRORE":
                    status = "error"
                    pred_label = "ERROR"
                else:
                    status = "ok"
                    pred_label = clean_and_map_nli_prediction(raw_pred)

                record = {
                    "id": rec_id,
                    "status": status,
                    "target_label": target_label,
                    "pred_label": pred_label,
                    "pred_raw": raw_pred,
                    "latency_seconds": round(latency, 4),
                }

                results.append(record)
                fmt2_data["results"] = results
                processed_ids.add(rec_id)

                # Scrittura Log Raw (.jsonl)
                raw_log_entry = {
                    "id": rec_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "task": "NLI",
                    "protocol_mode": protocol_mode,
                    "model_id": api_model,
                    "prompt_raw_sent": prompt_raw_text,
                    "response_raw_received": raw_pred,
                    "latency_seconds": round(latency, 4)
                }
                with open(raw_log_file, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(raw_log_entry, ensure_ascii=False) + "\n")

                curr_count = len(results)
                idx_str = f"{curr_count:>{pad_width}}"

                if VERBOSE:
                    print(f"[{idx_str}/{total_records}] ID: {rec_id} | Pred: {pred_label:<14} | Target: {target_label}")
                elif curr_count % STATUS_EVERY == 0 or curr_count == total_records:
                    elapsed_sec = time.time() - start_time
                    avg_lat = elapsed_sec / curr_count if curr_count > 0 else 0
                    print(f"⚡ Progresso [{api_model} | {protocol_mode}]: [{idx_str}/{total_records}] coppie elaborate | Lat. media: {avg_lat:.2f}s/record")

                # Checkpoint FMT2
                if curr_count % CHECKPOINT_EVERY == 0 or curr_count == total_records:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(fmt2_data, f, ensure_ascii=False, indent=2)

                time.sleep(0.02)

        except KeyboardInterrupt:
            print(f"\n🛑 Interruzione manuale durante [{api_model} - {protocol_mode.upper()}].")
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
            break

        elapsed_min = (time.time() - start_time) / 60.0
        print(f"\n✅ COMPLETATO ESPERIMENTO [{api_model} | {protocol_mode.upper()}] in {elapsed_min:.2f} minuti!\n")
        global_exp_idx += 1

print("🎉 TUTTI GLI ESPERIMENTI e-RTE-3-it SU DEEPINFRA SONO STATI ULTIMATI!")
