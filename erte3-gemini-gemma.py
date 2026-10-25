"""
================================================================================
SCRIPT BATCH COLAB OPENROUTER: BENCHMARK e-RTE-3-it (3-CLASS) - GEMMA 3 12B
================================================================================
Esegue la sequenza completa di esperimenti NLI su 'google/gemma-3-12b-it':
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
import pandas as pd
import requests
from google.colab import drive, userdata

# ==============================================================================
# 1. CONFIGURAZIONE ESPERIMENTI E AMBIENTE
# ==============================================================================
API_MODEL = "google/gemma-3-12b-it"
CHECKPOINT_EVERY = 64  # Salvataggio atomico FMT2
STATUS_EVERY = 64  # Messaggio di progresso a schermo
VERBOSE = False

TEMPERATURE = 0.0
MAX_TOKENS = 10

# Directory di lavoro dedicata a e-RTE-3-it
DRIVE_DIR = "/content/drive/MyDrive/erte3_eval"

# Dataset aumentato di input
DATASET_PATH = os.path.join(
    DRIVE_DIR, "e_rte3_it_BENCHMARK_FULL_3CLASS_UPTO8.csv"
)

# Modalita sperimentali
EVAL_MODES = [
    "0-shot",
    "0-shot-alt",
    "random-fixed",
    "random-variable",
    "semantic-minilm",
]
K_SHOTS_TO_TEST = [3, 5, 8]

OVERWRITE = False

# ==============================================================================
# 2. INIZIALIZZAZIONE OPENROUTER & GOOGLE DRIVE
# ==============================================================================
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

try:
  openrouter_api_key = userdata.get("OPENROUTER_API_KEY")
except Exception:
  openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

if not openrouter_api_key:
  openrouter_api_key = input("⚠️ Inserisci la tua OPENROUTER_API_KEY: ")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://colab.research.google.com/",
    "X-OpenRouter-Title": "e-RTE-3-it Benchmark Colab Gemma",
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


def build_nli_prompt(
    premise: str, hypothesis: str, mode: str, fewshot_ctx: str = ""
) -> str:
  """Costruisce la stringa di prompt esatta per il task NLI."""
  if mode == "0-shot":
    return (
        "Determina la relazione logica tra la Premessa e l'Ipotesi"
        " (Implicazione, Contraddizione, Neutro).\nRispondi SOLTANTO con una"
        f' parola.\n\nPremessa: "{premise}"\nIpotesi:'
        f' "{hypothesis}"\nRisposta:'
    )
  elif mode == "0-shot-alt":
    return (
        "Analizza la Premessa e l'Ipotesi seguenti e indica la relazione"
        " semantica scegliendo unicamente tra: Implicazione, Contraddizione,"
        " Neutro.\nRestituisci solo l'etichetta senza altri"
        f' commenti.\n\nPremessa: "{premise}"\nIpotesi:'
        f' "{hypothesis}"\nRelazione:'
    )
  else:
    return (
        "Determina la relazione logica tra la Premessa e l'Ipotesi"
        " (Implicazione, Contraddizione, Neutro).\nDi seguito trovi alcuni"
        f" esempi di riferimento per orientarti:\n\n{fewshot_ctx}\n\nOra"
        f' analizza questa coppia:\nPremessa: "{premise}"\nIpotesi:'
        f' "{hypothesis}"\nRispondi SOLTANTO con una parola (Implicazione,'
        " Contraddizione, Neutro).\nRisposta:"
    )


def predict_nli_gemma(prompt_text: str) -> tuple[str, float]:
  """Invia la richiesta a Gemma 3 12B via OpenRouter."""
  payload = {
      "model": API_MODEL,
      "messages": [{"role": "user", "content": prompt_text}],
      "temperature": TEMPERATURE,
      "max_tokens": MAX_TOKENS,
  }

  t0 = time.time()
  for attempt in range(1, 6):
    try:
      response = requests.post(
          url=OPENROUTER_URL,
          headers=HEADERS,
          data=json.dumps(payload),
          timeout=30,
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
      print(
          f"\n⚠️ [ERRORE OPENROUTER] Attempt {attempt}/5. Attesa {wait_time}s..."
          f" Errore: {e}"
      )
      time.sleep(wait_time)

  return "ERRORE", time.time() - t0


# ==============================================================================
# 4. CARICAMENTO DATASET
# ==============================================================================
if not os.path.exists(DATASET_PATH):
  raise FileNotFoundError(
      f"❌ Impossibile trovare il dataset NLI in: {DATASET_PATH}. Esegui prima"
      " lo script di generazione locale e-RTE-3-it."
  )

df_test = pd.read_csv(DATASET_PATH)
total_records = len(df_test)
pad_width = len(str(total_records))

print(
    f"📊 Dataset e-RTE-3-it Caricato: {total_records} coppie da {DATASET_PATH}\n"
)

# ==============================================================================
# 5. CICLO MASTER ESPERIMENTI IN FORMATO FMT2 + RAW LOG
# ==============================================================================
model_slug = API_MODEL.replace("/", "_").replace("-", "_").replace(".", "_")

experiments_list = []
for mode in EVAL_MODES:
  if mode in ["0-shot", "0-shot-alt"]:
    experiments_list.append((mode, 0, mode))
  else:
    for k in K_SHOTS_TO_TEST:
      protocol_name = f"{k}-shot-{mode}"
      experiments_list.append((mode, k, protocol_name))

print(
    f"🚀 AVVIO BENCHMARK e-RTE-3-it ({len(experiments_list)} CONFIGURAZIONI"
    f" TOTALI SU {API_MODEL})\n"
)

for exp_index, (eval_mode, k_shots, protocol_mode) in enumerate(
    experiments_list, 1
):
  checkpoint_file = os.path.join(
      DRIVE_DIR, f"checkpoint-fmt2-{protocol_mode}-erte3-{model_slug}.json"
  )
  raw_log_file = os.path.join(
      DRIVE_DIR, f"log-raw-{protocol_mode}-erte3-{model_slug}.jsonl"
  )

  print("=" * 80)
  print(
      f"▶️ ESPERIMENTO {exp_index}/{len(experiments_list)}:"
      f" [{protocol_mode.upper()}] su [{API_MODEL}]"
  )
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
      print(
          f"🔄 RIPRESA AUTOMATICA: {len(processed_ids)}/{total_records} coppie"
          " già elaborate."
      )
    except Exception as e:
      print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero.")

  if len(results) == total_records:
    print(
        f"✅ Esperimento [{protocol_mode.upper()}] già COMPLETATO al 100%."
        " Salto al successivo.\n"
    )
    continue

  fmt2_data = {
      "metadata": {
          "model": {"provider": "openrouter", "model_id": API_MODEL},
          "protocol": {"task": "NLI", "mode": protocol_mode},
          "experiment": {
              "eval_mode": eval_mode,
              "k_shots": k_shots,
              "temperature": TEMPERATURE,
              "max_tokens": MAX_TOKENS,
              "status_every": STATUS_EVERY,
          },
          "dataset": {"path": DATASET_PATH, "total_records": total_records},
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

      prompt_raw_text = build_nli_prompt(
          premise_text, hypothesis_text, eval_mode, k_shot_ctx
      )
      raw_pred, latency = predict_nli_gemma(prompt_raw_text)

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
          "model_id": API_MODEL,
          "prompt_raw_sent": prompt_raw_text,
          "response_raw_received": raw_pred,
          "latency_seconds": round(latency, 4),
      }
      with open(raw_log_file, "a", encoding="utf-8") as rf:
        rf.write(json.dumps(raw_log_entry, ensure_ascii=False) + "\n")

      curr_count = len(results)
      idx_str = f"{curr_count:>{pad_width}}"

      if VERBOSE:
        print(
            f"[{idx_str}/{total_records}] ID: {rec_id} | Pred:"
            f" {pred_label:<14} | Target: {target_label}"
        )
      elif curr_count % STATUS_EVERY == 0 or curr_count == total_records:
        elapsed_sec = time.time() - start_time
        avg_lat = elapsed_sec / curr_count if curr_count > 0 else 0
        print(
            f"⚡ Progresso e-RTE-3-it: [{idx_str}/{total_records}] coppie"
            f" elaborate | Lat. media: {avg_lat:.2f}s/record"
        )

      # Checkpoint FMT2
      if curr_count % CHECKPOINT_EVERY == 0 or curr_count == total_records:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
          json.dump(fmt2_data, f, ensure_ascii=False, indent=2)

      time.sleep(0.05)

  except KeyboardInterrupt:
    print(f"\n🛑 Interruzione manuale durante [{protocol_mode.upper()}].")
    with open(checkpoint_file, "w", encoding="utf-8") as f:
      json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
    break

  elapsed_min = (time.time() - start_time) / 60.0
  print(
      f"\n✅ COMPLETATO ESPERIMENTO e-RTE-3-it [{protocol_mode.upper()}] in"
      f" {elapsed_min:.2f} minuti!\n"
  )

print(f"🎉 TUTTI GLI ESPERIMENTI e-RTE-3-it PER {API_MODEL} SONO STATI ULTIMATI!")
