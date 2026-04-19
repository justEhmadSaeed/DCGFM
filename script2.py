import os
import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading SBERT...")
model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")
print("SBERT Loaded\n")

# Collect everything here, write once at the end
all_rows = []  # list of (dataset, label, embedding_vector)

def process(dataset_name, texts):
    """Encode texts, print sample, store for file output."""
    print(f"===== {dataset_name} =====")
    embeddings = model.encode(texts)
    print(f"{dataset_name} Shape:", embeddings.shape)
    print("Sample:", embeddings[0][:10])
    print("=" * (len(dataset_name) + 12), "\n")

    for text, emb in zip(texts, embeddings):
        all_rows.append((dataset_name, text, emb))


# ---------- 1. CORA ----------
cora_file = "OFA/data/single_graph/Cora/categories.csv"
if os.path.exists(cora_file):
    texts = []
    with open(cora_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            texts.append(line.strip())
            if i > 50:
                break
    process("CORA", texts)


# ---------- 2. PUBMED ----------
pubmed_file = "OFA/data/single_graph/Pubmed/categories.txt"
if os.path.exists(pubmed_file):
    texts = []
    with open(pubmed_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            texts.append(line.strip())
            if i > 50:
                break
    process("PUBMED", texts)


# ---------- 3. FB15K237 ----------
kg_file = "OFA/data/KG/FB15K237/train.txt"
if os.path.exists(kg_file):
    texts = []
    with open(kg_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            head, rel, tail = line.strip().split("\t")
            texts.extend([head, rel, tail])
            if i > 50:
                break
    process("FB15K237", texts)


# ---------- 4. WN18RR ----------
kg_file = "OFA/data/KG/WN18RR/train.txt"
if os.path.exists(kg_file):
    texts = []
    with open(kg_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            head, rel, tail = line.strip().split("\t")
            texts.extend([head, rel, tail])
            if i > 50:
                break
    process("WN18RR", texts)


# ---------- 5. CHEMMOL ----------
# Structure: { dataset_name -> { class_id -> [desc_str, ...] } }
# Previously we passed raw dict objects to SBERT; now we extract actual text.
mol_file = "OFA/data/chemmol/mol_label_desc.json"
if os.path.exists(mol_file):
    with open(mol_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []  # (dataset_name, class_id, desc_index, text)
    for ds_name, class_dict in data.items():
        for cls_id, desc_list in class_dict.items():
            for desc_idx, text in enumerate(desc_list):
                records.append((ds_name, cls_id, desc_idx, text))

    print(f"CHEMMOL total label descriptions across all datasets/classes: {len(records)}")

    # Embed the first description per (dataset, class) — deduplicated entry point
    # matching what OFA uses as the canonical class node text.
    seen = set()
    texts_chemmol = []
    meta_chemmol  = []
    for ds_name, cls_id, desc_idx, text in records:
        key = (ds_name, cls_id)
        if key not in seen:
            seen.add(key)
            texts_chemmol.append(text)
            meta_chemmol.append(f"{ds_name}/class_{cls_id}")

    print(f"Unique (dataset, class) entries: {len(texts_chemmol)}")
    process("CHEMMOL", texts_chemmol)


# =====================================================
# WRITE OUTPUTS
# =====================================================

# --- Readable TXT ---
with open("embeddings.txt", "w", encoding="utf-8") as f:
    for dataset, label, emb in all_rows:
        f.write(f"[{dataset}] {label}\n")
        f.write(np.array2string(emb, separator=", ", max_line_width=np.inf))
        f.write("\n\n")

# --- Structured CSV ---
if all_rows:
    dim = len(all_rows[0][2])
    with open("embeddings.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label"] + [f"dim_{i}" for i in range(dim)])
        for dataset, label, emb in all_rows:
            writer.writerow([dataset, label] + emb.tolist())

print(f"Saved {len(all_rows)} embeddings to embeddings.txt and embeddings.csv")
print("Done.")