"""
================================================================================
DASHBOARD DI VALUTAZIONE ARRICCHITA BENCHMARK NLI (e-RTE-3-it)
================================================================================
Analizza tutti i checkpoint FMT2 presenti in '/content/drive/MyDrive/erte3_eval'
e produce una dashboard completa con Accuracy, Macro F1, Latenza, Throughput e Provider.
================================================================================
"""

import os
import json
import glob
import pandas as pd
from google.colab import drive
from sklearn.metrics import accuracy_score, f1_score

# ==============================================================================
# CONFIGURAZIONE PERCORSI
# ==============================================================================
DRIVE_DIR = "/content/drive/MyDrive/erte3_eval"
LABELS = ["Implicazione", "Contraddizione", "Neutro"]

drive.mount("/content/drive", force_remount=False)

def stampa_dashboard_arricchita(summary_rows):
    """Stampa a schermo la dashboard avanzata con colonne allineate, Macro F1 e separatori a pipe (|)."""
    if not summary_rows:
        print("⚠️ Nessun dato da stampare.")
        return

    print("\n" + "=" * 154)
    print(f"| {'MODELLO':<34} | {'PROVIDER':<10} | {'PROTOCOLLO':<18} | {'STATO':<10} | {'ACCURACY':<9} | {'MACRO F1':<9} | {'LATENZA':<9} | {'REC/SEC':<8} |")
    print("=" * 154)

    for r in sorted(summary_rows, key=lambda x: x["Accuracy_Raw"], reverse=True):
        modello = r["Modello"][:32]
        provider = r["Provider"][:8]
        protocollo = r["Modalita"][:16]
        stato = r["Stato"][:8]
        accuracy = r["Accuracy"]
        macro_f1 = r["Macro F1"]
        lat_media = r["Sec/Rec"]
        rec_sec = f"{r['Sec_Sec']:.1f}" if r["Sec_Sec"] > 0 else "-"

        print(f"| {modello:<34} | {provider:<10} | {protocollo:<18} | {stato:<10} | {accuracy:<9} | {macro_f1:<9} | {lat_media:<9} | {rec_sec:<8} |")

    print("=" * 154 + "\n")

def generate_enriched_nli_dashboard():
    pattern = os.path.join(DRIVE_DIR, "checkpoint-fmt2-*-erte3-*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠️ Nessun file checkpoint trovato matching il pattern: {pattern}")
        return

    summary_rows = []

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            results = data.get("results", [])

            model_id = metadata.get("model", {}).get("model_id", "Unknown")
            provider = metadata.get("model", {}).get("provider", "Unknown")
            protocol = metadata.get("protocol", {}).get("mode", "Unknown")
            total_records = metadata.get("dataset", {}).get("total_records", 800)

            completed_count = len(results)
            progress_pct = (completed_count / total_records) * 100 if total_records > 0 else 0

            status_str = "COMPLETATO" if completed_count == total_records else f"{progress_pct:.1f}%"

            if completed_count == 0:
                continue

            df_res = pd.DataFrame(results)
            df_valid = df_res[df_res["status"] == "ok"]

            if len(df_valid) == 0:
                acc, f1_macro, avg_lat, rec_per_sec = 0.0, 0.0, 0.0, 0.0
            else:
                y_true = df_valid["target_label"]
                y_pred = df_valid["pred_label"]

                acc = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
                avg_lat = df_valid["latency_seconds"].mean() if "latency_seconds" in df_valid.columns else 0.0
                rec_per_sec = (1.0 / avg_lat) if avg_lat > 0 else 0.0

            summary_rows.append({
                "Modello": model_id,
                "Provider": provider.upper(),
                "Modalita": protocol,
                "Stato": status_str,
                "Progresso": f"{completed_count}/{total_records}",
                "Accuracy": f"{acc * 100:.2f}%",
                "Accuracy_Raw": acc,
                "Macro F1": f"{f1_macro:.4f}",
                "Sec/Rec": f"{avg_lat:.2f}s" if avg_lat > 0 else "-",
                "Sec_Sec": rec_per_sec,
            })

        except Exception as e:
            print(f"⚠️ Errore durante l'analisi del file {os.path.basename(filepath)}: {e}")

    if not summary_rows:
        print("⚠️ Nessun risultato valido analizzato.")
        return

    # Visualizzazione classica Pandas + Tabella arricchita con pipe
    df_summary = pd.DataFrame(summary_rows).drop(columns=["Accuracy_Raw", "Sec_Sec"])
    print("\n" + "=" * 125)
    print(f"| DASHBOARD BENCHMARK NLI (e-RTE-3-it) - {DRIVE_DIR:<52} |")
    print("=" * 125)
    print(df_summary.to_string(index=False))
    print("=" * 125 + "\n")

    stampa_dashboard_arricchita(summary_rows)

if __name__ == "__main__":
    generate_enriched_nli_dashboard()
