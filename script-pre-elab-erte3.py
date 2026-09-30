import os
import xml.etree.ElementTree as ET
from pathlib import Path


def load_rte_xml_verbose(filepath):
  print(f"\n" + "=" * 60)
  print(f"🔍 ANALISI DIAGNOSTICA DETTAGLIATA: {filepath}")
  print("=" * 60)

  # 1. Verifica Esistenza
  path = Path(filepath)
  if not path.exists():
    print(f"❌ ERRORE CRITICO: Il file NON esiste al percorso specificato!")
    print(f"   Percorso cercato: {path.absolute()}")
    return []

  # 2. Controllo Dimensione File
  size_bytes = os.path.getsize(filepath)
  print(f"📄 Dimensione file: {size_bytes} byte ({size_bytes / 1024:.2f} KB)")
  if size_bytes == 0:
    print("❌ ERRORE CRITICO: Il file su Drive è VUOTO (0 byte)!")
    return []

  # 3. Lettura flessibile con rimozione BOM e spazi spuri
  with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    raw_content = f.read()

  # Sanitizzazione dello stream
  content = raw_content.strip()
  if "\ufeff" in content:
    content = content.replace("\ufeff", "")

  # 4. Parsing XML
  try:
    root = ET.fromstring(content)
    print(f"✅ Parsing XML eseguito con successo.")
    print(f"📌 Root Tag rilevato: <{root.tag}>")
  except ET.ParseError as e:
    print(f"⚠️ Errore durante ET.fromstring: {e}")
    # Tentativo di recupero se manca il tag di chiusura finale
    if not content.endswith(">"):
      content += "</entailment-corpus>"
    try:
      root = ET.fromstring(content)
      print("✅ Recupero XML completato con successo!")
    except Exception as ex:
      print(f"❌ Impossibile interpretare la struttura XML: {ex}")
      return []

  # 5. Ispezione dei Tag interni per trovare le coppie
  all_tags = set(elem.tag for elem in root.iter())
  print(f"🏷️  Tutti i tag presenti nel documento: {all_tags}")

  # Ricerca agnostica (gestisce maiuscole/minuscole e namespace)
  pairs_elements = [
      elem for elem in root.iter() if elem.tag.lower().endswith("pair")
  ]
  print(f"🎯 Coppie <pair> estratte: {len(pairs_elements)}")

  extracted_dataset = []
  for idx, pair in enumerate(pairs_elements):
    # Ricerca agnostica dei sottonodi 't' ed 'h'
    t_elem = next(
        (child for child in pair if child.tag.lower().endswith("t")), None
    )
    h_elem = next(
        (child for child in pair if child.tag.lower().endswith("h")), None
    )

    t_text = (
        t_elem.text.strip()
        if (t_elem is not None and t_elem.text)
        else ""
    )
    h_text = (
        h_elem.text.strip()
        if (h_elem is not None and h_elem.text)
        else ""
    )

    extracted_dataset.append({
        "id": pair.get("id", str(idx + 1)),
        "entailment": pair.get("entailment", ""),
        "task": pair.get("task", ""),
        "text": t_text,
        "hypothesis": h_text,
    })

  # Stampa di verifica del primo elemento
  if extracted_dataset:
    print("  --> Esempio Primo Record Caricato:")
    print(f"      ID: {extracted_dataset[0]['id']}")
    print(f"      Text: {extracted_dataset[0]['text'][:80]}...")
    print(f"      Hypothesis: {extracted_dataset[0]['hypothesis'][:80]}...")
    print(f"      Entailment: {extracted_dataset[0]['entailment']}")
  else:
    print("⚠️ NESSUN RECORD ESTRATTO. Verificare l'output dei tag sopra.")

  return extracted_dataset


# 🚀 ESECUZIONE SUI FILE IN DRIVE
DEV_PATH = "/content/drive/MyDrive/erte3_eval/DEV.xml"
TEST_PATH = "/content/drive/MyDrive/erte3_eval/TEST.xml"

dev_records = load_rte_xml_verbose(DEV_PATH)
test_records = load_rte_xml_verbose(TEST_PATH)

print(
    f"\n🟢 RIEPILOGO FINALE: DEV = {len(dev_records)} record | TEST ="
    f" {len(test_records)} record"
)
