"""
script.py — Analysis of haitengzhao/molecule_property_instruction dataset
==========================================================================
Dataset schema (one row):
  graph          : SMILES string  (the molecule)
  text           : list[str]      (natural-language task / assay descriptions)
  label          : str            (Yes / No — the property answer)
  dataset_name   : str            (bace, hiv, tox21, …)
  task_index     : str
  molecule_index : str
  split          : str            (train / test / val)

Pipeline (mirrors DCGFM / OFA):
  1. SMILES → molecular graph  (atoms = nodes, bonds = edges)
  2. Each atom → enriched text description → SBERT → h_v  (semantic node embedding)
  3. Task description text        → SBERT → h_task (subgraph-level semantic)
  4. Hard-pruning informativity score = distance of h_task from distribution centre
     computed over ALL benchmark datasets (scanning every HuggingFace split).
"""

# ── imports ──────────────────────────────────────────────────────────────────
import re, collections
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer

# ── helpers: lightweight SMILES → graph (no rdkit needed) ────────────────────
BOND_CHARS = {"-": "single", "=": "double", "#": "triple", ":": "aromatic"}

# Standard organic valences for implicit-H estimation.
# Without RDKit, the parser only captures explicit bracket H counts; this table
# fills the gap so degree-based atom descriptions are chemically accurate.
_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "F": 1,
            "Cl": 1, "Br": 1, "I": 1, "P": 3, "Si": 4, "B": 3}


def smiles_to_graph(smiles: str):
    """
    Minimal SMILES parser → returns (nodes, edges).
      nodes : list of atom dicts  {idx, symbol, aromatic, charge, hcount}
      edges : list of bond dicts  {src, dst, bond_type}
    Handles rings, branches, charges, H-counts (no stereo/isotope needed here).
    Note: implicit hydrogens are not counted; only explicit bracket H's appear.
    """
    nodes = []
    edges = []
    stack = []       # branch stack: (prev_node_idx, pending_bond)
    ring_open = {}   # ring-digit → (node_idx, bond_type)
    prev = None
    pending_bond = "single"
    i = 0

    def add_atom(symbol, aromatic=False, charge=0, hcount=0):
        idx = len(nodes)
        nodes.append({"idx": idx, "symbol": symbol, "aromatic": aromatic,
                       "charge": charge, "hcount": hcount})
        return idx

    while i < len(smiles):
        c = smiles[i]

        if c in BOND_CHARS:
            pending_bond = BOND_CHARS[c]
            i += 1
            continue

        if c == "(":
            stack.append((prev, pending_bond))
            i += 1
            continue

        if c == ")":
            prev, pending_bond = stack.pop()
            i += 1
            continue

        # ring closure
        if c.isdigit() or (c == "%" and i + 2 < len(smiles)):
            if c == "%":
                digit = smiles[i+1:i+3]; i += 3
            else:
                digit = c; i += 1
            if digit in ring_open:
                src, bt = ring_open.pop(digit)
                bt = bt if bt != "single" else pending_bond
                edges.append({"src": src, "dst": prev, "bond_type": bt})
            else:
                ring_open[digit] = (prev, pending_bond)
            pending_bond = "single"
            continue

        # bracketed atom  [NH3+]  [C@@H]
        if c == "[":
            j = smiles.index("]", i)
            inner = smiles[i+1:j]; i = j + 1
            m = re.match(r"\d*([A-Z][a-z]?|[a-z])", inner)
            sym = m.group(1) if m else "?"
            aromatic = sym.islower(); sym = sym.capitalize()
            charge = inner.count("+") - inner.count("-")
            hm = re.search(r"H(\d*)", inner)
            hcount = int(hm.group(1)) if hm and hm.group(1) else (1 if hm else 0)
            idx = add_atom(sym, aromatic, charge, hcount)
            if prev is not None:
                edges.append({"src": prev, "dst": idx, "bond_type": pending_bond})
            prev = idx; pending_bond = "single"
            continue

        # two-char organic  Cl  Br
        if c in "CB" and i + 1 < len(smiles) and smiles[i:i+2] in ("Cl", "Br"):
            sym = smiles[i:i+2]; i += 2
            idx = add_atom(sym)
            if prev is not None:
                edges.append({"src": prev, "dst": idx, "bond_type": pending_bond})
            prev = idx; pending_bond = "single"
            continue

        # single-char atom
        if c.isalpha():
            aromatic = c.islower(); sym = c.upper()
            idx = add_atom(sym, aromatic)
            bt = "aromatic" if aromatic else pending_bond
            if prev is not None:
                edges.append({"src": prev, "dst": idx, "bond_type": bt})
            prev = idx; pending_bond = "single"; i += 1
            continue

        i += 1  # skip stereo (@, /) and dots

    return nodes, edges


def add_implicit_h(nodes: list, edges: list) -> None:
    """
    Estimate implicit hydrogen count for each atom using standard valence rules
    and write it back into each node dict as 'impl_h' (in-place).

    Without RDKit the parser only records explicit bracket-H counts, so atoms
    like the bare 'c' or 'C' in a SMILES string always show hcount=0.  This
    makes degree-based descriptions wrong and collapses embeddings for atoms
    that should be distinguishable (e.g. CH2 vs CH3 vs quaternary C).

    Formula:  impl_h = max(0, typical_valence + formal_charge − graph_degree − explicit_h)
    """
    degree = collections.Counter()
    for e in edges:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
    for a in nodes:
        val = _VALENCE.get(a["symbol"])
        if val is None:
            a["impl_h"] = 0
            continue
        a["impl_h"] = max(0, val + a["charge"] - degree[a["idx"]] - a["hcount"])


def find_ring_atoms(nodes: list, edges: list) -> set:
    """
    Return the set of atom indices that lie in at least one ring.
    Uses iterative DFS back-edge detection on the undirected molecular graph.
    A back edge (u → ancestor v already in the DFS stack) proves a cycle; all
    nodes on the path from u back to v through parent pointers are in that ring.
    """
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e["src"]].append(e["dst"])
        adj[e["dst"]].append(e["src"])

    in_ring = set()
    color   = [0] * n   # 0=unvisited  1=in DFS stack  2=finished
    parent  = [-1] * n

    def dfs(start: int):
        # Iterative DFS using an explicit stack of (node, neighbour-iterator) pairs
        stack = [(start, iter(adj[start]))]
        color[start] = 1
        while stack:
            v, nbrs = stack[-1]
            try:
                nb = next(nbrs)
                if color[nb] == 0:
                    parent[nb] = v
                    color[nb] = 1
                    stack.append((nb, iter(adj[nb])))
                elif color[nb] == 1 and nb != parent[v]:
                    # back edge → trace cycle through parent pointers
                    cur = v
                    while cur != nb:
                        in_ring.add(cur)
                        cur = parent[cur]
                    in_ring.add(nb)
            except StopIteration:
                color[v] = 2
                stack.pop()

    for i in range(n):
        if color[i] == 0:
            dfs(i)
    return in_ring


def atom_to_text(atom: dict, nodes: list, edges: list, ring_atoms: set) -> str:
    """
    Enriched natural-language node description for SBERT encoding.

    Over the bare OFA baseline ("This is a C atom"), this version adds:
      - Ring membership (aromatic vs. non-aromatic)
      - Explicit degree
      - Neighbour element summary
      - Double / triple bond counts
      - Hybridisation hint (sp / sp2 / sp3)

    These extras reduce the cosine-similarity collapse observed when atoms of
    the same element type receive almost-identical embeddings, giving GIN a more
    distinguishable starting point before message passing.
    """
    sym = atom["symbol"]
    idx = atom["idx"]

    # Gather neighbour info directly from the edge list (O(|E|) per call)
    nb_bonds = [
        (e["dst"] if e["src"] == idx else e["src"], e["bond_type"])
        for e in edges if idx in (e["src"], e["dst"])
    ]
    nb_syms = [nodes[nb]["symbol"] for nb, _ in nb_bonds]
    bt_here = [bt for _, bt in nb_bonds]
    degree  = len(nb_bonds)
    bt_cnt  = collections.Counter(bt_here)

    # Opening phrase varies by chemical context so SBERT sees meaningfully
    # different sentence starts rather than the same "This is a C atom" prefix.
    if atom["aromatic"]:
        parts = [f"Aromatic {sym} in a ring"]
    elif idx in ring_atoms:
        parts = [f"Non-aromatic ring {sym}"]
    else:
        parts = [f"Chain {sym} atom"]

    # Formal charge
    if atom["charge"] > 0:
        parts.append(f"with a +{atom['charge']} formal charge")
    elif atom["charge"] < 0:
        parts.append(f"with a {atom['charge']} formal charge")

    # Total H = explicit bracket H + estimated implicit H (from valence table)
    total_h = atom["hcount"] + atom.get("impl_h", 0)
    if total_h > 0:
        parts.append(f"bonded to {total_h} hydrogen(s)")

    # Degree
    parts.append(f"with degree {degree}")

    # Neighbour element summary
    if nb_syms:
        elem_cnt = collections.Counter(nb_syms)
        nb_str   = ", ".join(f"{cnt} {el}" for el, cnt in elem_cnt.most_common())
        parts.append(f"connected to {nb_str}")

    # Bond-type annotations
    if bt_cnt["double"]:
        parts.append(f"with {bt_cnt['double']} double bond(s)")
    if bt_cnt["triple"]:
        parts.append(f"with {bt_cnt['triple']} triple bond(s)")

    # Hybridisation hint (coarse but informative for SBERT)
    if atom["aromatic"]:
        parts.append("sp2 hybridised")
    elif bt_cnt["triple"]:
        parts.append("sp hybridised")
    elif bt_cnt["double"]:
        parts.append("sp2 hybridised")
    else:
        parts.append("sp3 hybridised")

    return ", ".join(parts) + "."


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Loading dataset: haitengzhao/molecule_property_instruction")
print("=" * 65)
ds = load_dataset("haitengzhao/molecule_property_instruction", trust_remote_code=True)

print(f"\nSplits : {list(ds.keys())}")
for split_name, split_ds in ds.items():
    print(f"  {split_name:28s}: {len(split_ds):>7,} rows")

# Draw 500 rows from every split (shuffled, fixed seed) and concatenate.
# This gives cross-dataset coverage instead of bace-only analysis.
SAMPLE_PER_SPLIT = 500
split_samples = [
    ds[sname].shuffle(seed=42).select(range(min(SAMPLE_PER_SPLIT, len(ds[sname]))))
    for sname in ds.keys()
]
sample = concatenate_datasets(split_samples)
print(f"\nSampling {SAMPLE_PER_SPLIT} rows from each of {len(ds)} splits "
      f"→ {len(sample):,} rows total.")
print(f"Features:\n  {sample.features}")

# ── SECTION 1: Dataset composition ───────────────────────────────────────────
print("\n" + "=" * 65)
print(f"SECTION 1 — Dataset composition ({len(sample):,} rows across all splits)")
print("=" * 65)

label_counts = collections.Counter()
ds_counts    = collections.Counter()
split_counts = collections.Counter()
# Batch-fetch columns as lists — one call each, no Python row iteration.
label_counts.update(sample["label"])
ds_counts.update(sample["dataset_name"])
split_counts.update(sample["split"])

print(f"\nLabel distribution : {dict(label_counts)}")
print(f"Split distribution : {dict(split_counts)}")
print(f"\nRows per molecular benchmark:")
for k, v in ds_counts.most_common():
    print(f"  {k:28s}: {v:6,}")

# ── SECTION 2: Raw row schema ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2 — Sample rows (raw fields)")
print("=" * 65)
for i in range(3):
    row = sample[i]
    print(f"\n--- Row {i} ---")
    print(f"  dataset_name   : {row['dataset_name']}")
    print(f"  split          : {row['split']}")
    print(f"  label          : {row['label']}")
    print(f"  molecule_index : {row['molecule_index']}")
    print(f"  task_index     : {row['task_index']}")
    print(f"  graph (SMILES) : {row['graph'][:80]}")
    print(f"  # text variants: {len(row['text'])}")
    print(f"  text[0][:220]  :\n    {row['text'][0][:220]}")

# ── SECTION 3: SMILES → Molecular subgraph ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3 — SMILES → Molecular Subgraph")
print("=" * 65)
print("""
In OFA/DCGFM, every molecule is treated as a subgraph:
  • Atoms   = nodes  (each carries an enriched text description as its node feature)
  • Bonds   = edges  (typed: single / double / triple / aromatic)
  • SBERT encodes each atom's text → h_v  (initial GIN node embedding)
  • GIN aggregates h_v across the molecular graph → h_g  (subgraph embedding)
  • h_g feeds into the DeepSVDD objective for hard-pruning scoring
""")

for sample_idx in range(3):
    row = sample[sample_idx]
    smiles = row["graph"]
    nodes, edges = smiles_to_graph(smiles)
    add_implicit_h(nodes, edges)
    ring_atoms = find_ring_atoms(nodes, edges)
    print(f"--- Molecule {sample_idx}  ({row['dataset_name']}, label={row['label']}) ---")
    print(f"  SMILES : {smiles}")
    print(f"  Atoms  : {len(nodes)}   Bonds: {len(edges)}")
    deg = collections.Counter()
    for e in edges:
        deg[e["src"]] += 1; deg[e["dst"]] += 1
    if deg:
        dv = list(deg.values())
        print(f"  Degree : min={min(dv)}  max={max(dv)}  mean={np.mean(dv):.2f}")
    bond_types = collections.Counter(e["bond_type"] for e in edges)
    print(f"  Bond types : {dict(bond_types)}")
    elem = collections.Counter(a["symbol"] for a in nodes)
    print(f"  Elements   : {dict(elem.most_common())}")
    print(f"  Ring atoms : {len(ring_atoms)} / {len(nodes)}  "
          f"({100*len(ring_atoms)/max(len(nodes),1):.0f}%)")
    print()

# Detailed atom + edge dump for molecule 0
row0 = sample[0]
nodes0, edges0 = smiles_to_graph(row0["graph"])
add_implicit_h(nodes0, edges0)
ring_atoms0 = find_ring_atoms(nodes0, edges0)
print(f"Detailed atom list for molecule 0 (SMILES: {row0['graph'][:60]}):")
for a in nodes0:
    ring_flag = "ring" if a["idx"] in ring_atoms0 else "    "
    total_h = a["hcount"] + a.get("impl_h", 0)
    print(f"  [{a['idx']:2d}] {a['symbol']:3s}  aromatic={str(a['aromatic']):5s}  "
          f"charge={a['charge']:+d}  H={total_h}(expl={a['hcount']},impl={a.get('impl_h',0)})  {ring_flag}")

print(f"\nBond list (edges):")
for e in edges0:
    print(f"  {e['src']:2d} --{e['bond_type']:8s}--> {e['dst']}")

# ── SECTION 4: Atom → enriched text description ───────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4 — Atom Text Descriptions (enriched node text attributes)")
print("=" * 65)
print("""
Enriched over the OFA baseline: now includes neighbour element types, degree,
hybridisation hint, and ring membership.  These extras reduce the cosine-
similarity collapse when SBERT encodes identical-element atoms.
""")
for a in nodes0:
    print(f"  Atom {a['idx']:2d}: {atom_to_text(a, nodes0, edges0, ring_atoms0)}")

# ── SECTION 5: SBERT encoding ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5 — SBERT Encoding")
print("=" * 65)
print("\nLoading SBERT (multi-qa-distilbert-cos-v1)...")
sbert = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")
print("Loaded.\n")

# Collect one canonical task description per benchmark dataset.
# Strategy: grab row 0 from every HuggingFace split (13 fast single-row
# accesses) — this handles the common layout where each split = one dataset.
# Then scan the shuffled sample to catch any remaining datasets not yet seen
# (handles train/val/test layouts where multiple datasets share a split).
task_texts = {}
for sname in ds.keys():
    row = ds[sname][0]
    dn  = row["dataset_name"]
    if dn not in task_texts:
        task_texts[dn] = row["text"][0] if isinstance(row["text"], list) else row["text"]

for dn, text in zip(sample["dataset_name"], sample["text"]):
    if dn not in task_texts:
        task_texts[dn] = text[0] if isinstance(text, list) else text

dataset_names  = list(task_texts.keys())
task_desc_list = [task_texts[d] for d in dataset_names]

# 5a — task-level (subgraph-level) embeddings
print(f"Encoding {len(dataset_names)} task/assay descriptions "
      f"(one per benchmark dataset, from all {len(ds)} splits + sample)...")
task_embs = sbert.encode(task_desc_list, normalize_embeddings=True,
                         show_progress_bar=True)
print(f"\nTask embedding matrix : {task_embs.shape}  "
      f"(datasets × SBERT dims)")
print(f"All unit-normalised   : mean norm = "
      f"{np.linalg.norm(task_embs, axis=1).mean():.4f}")

print(f"\nSample — first 10 dims of each task embedding:")
for dn, emb in zip(dataset_names, task_embs):
    print(f"  {dn:20s}: [{', '.join(f'{x:6.3f}' for x in emb[:10])}]")

# 5b — atom-level (node-level) embeddings for molecule 0
atom_texts_mol0 = [atom_to_text(a, nodes0, edges0, ring_atoms0) for a in nodes0]
print(f"\nEncoding {len(atom_texts_mol0)} enriched atom descriptions for molecule 0...")
atom_embs_mol0 = sbert.encode(atom_texts_mol0, normalize_embeddings=True)
print(f"Atom embedding matrix : {atom_embs_mol0.shape}  (atoms × SBERT dims)")

print(f"\nAtom-level embedding preview (first 8 dims):")
for a, emb in zip(nodes0[:6], atom_embs_mol0[:6]):
    print(f"  [{a['idx']:2d}] {a['symbol']:3s}: "
          f"[{', '.join(f'{x:6.3f}' for x in emb[:8])}]")

# Pairwise similarity — enriched descriptions should reduce the redundancy
# vs. the bare-baseline mean of ~0.879 reported in the original analysis
atom_sim = atom_embs_mol0 @ atom_embs_mol0.T
n_atoms = len(nodes0)
off = atom_sim[~np.eye(n_atoms, dtype=bool)]
print(f"\nPairwise cosine sim between atom embeddings in molecule 0:")
print(f"  mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}")
print("  (Enriched descriptions should show lower mean and wider spread than the")
print("   bare-baseline ~0.879 mean — a sign of reduced semantic redundancy)")

# ── SECTION 6: DCGFM hard-pruning score ──────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6 — DCGFM Hard-Pruning Informativity Score")
print("=" * 65)
print(f"""
In DCGFM's hard-pruning step:
  c   = mean of ALL subgraph embeddings   (distribution centre)
  s_i = ||h_g_i - c||²                   (informativity score)
  → subgraphs with LOW  s_i are near the centre → redundant → pruned
  → subgraphs with HIGH s_i are diverse/informative → kept

Using {len(dataset_names)} task-level SBERT embeddings (one per benchmark dataset).
""")

centre    = task_embs.mean(axis=0)
distances = np.linalg.norm(task_embs - centre, axis=1) ** 2

print(f"Distribution centre norm : {np.linalg.norm(centre):.4f}")
print(f"\nInformativity scores  (s_i = ||h_task - c||²), highest first:")
score_pairs = sorted(zip(distances, dataset_names), reverse=True)
for score, dn in score_pairs:
    print(f"  {dn:28s}: {score:.6f}")

print(f"\nPruning decisions at different α (hard pruning ratio):")
for alpha in [0.3, 0.5, 0.7]:
    threshold = np.percentile(distances, alpha * 100)
    kept   = [dn for s, dn in score_pairs if s > threshold]
    pruned = [dn for s, dn in score_pairs if s <= threshold]
    print(f"  α={alpha:.1f} → keep {len(kept):2d}, prune {len(pruned):2d}")
    print(f"    kept  : {kept}")
    print(f"    pruned: {pruned}")

# Most similar task-description pairs (highest pruning candidates)
sim_matrix = task_embs @ task_embs.T
n = len(dataset_names)
print(f"\nMost similar task-description pairs (highest pruning candidates):")
pairs = []
for i in range(n):
    for j in range(i+1, n):
        pairs.append((sim_matrix[i, j], dataset_names[i], dataset_names[j]))
pairs.sort(reverse=True)
for sim, a, b in pairs[:6]:
    print(f"  {a:20s} ↔ {b:20s}  cos_sim={sim:.4f}")

# ── SECTION 7: Molecular graph size statistics ────────────────────────────────
print("\n" + "=" * 65)
print(f"SECTION 7 — Molecular Graph Size Statistics ({len(sample):,} rows, all splits)")
print("=" * 65)

node_counts = []; edge_counts = []
per_ds_nodes = collections.defaultdict(list)

for row in sample:
    ns, es = smiles_to_graph(row["graph"])
    add_implicit_h(ns, es)
    node_counts.append(len(ns))
    edge_counts.append(len(es))
    per_ds_nodes[row["dataset_name"]].append(len(ns))

nc = np.array(node_counts); ec = np.array(edge_counts)
print(f"\nAll molecules (n={len(nc)}):")
print(f"  Nodes (atoms): min={nc.min()}  max={nc.max()}  "
      f"mean={nc.mean():.1f}  median={np.median(nc):.0f}")
print(f"  Edges (bonds): min={ec.min()}  max={ec.max()}  "
      f"mean={ec.mean():.1f}  median={np.median(ec):.0f}")
print(f"\n  (Paper Table 4 reports ~25.9 avg nodes, ~55.9 avg edges for ChEMBL.)")

print(f"\nMean atoms per benchmark (sampled rows):")
for dn, ns_list in sorted(per_ds_nodes.items()):
    arr = np.array(ns_list)
    print(f"  {dn:28s}: mean={arr.mean():.1f}  min={arr.min()}  max={arr.max()}")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
