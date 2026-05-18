# BEACON: Benchmark for Entity Recognition Across Cybersecurity sources with unified ONtology

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.x-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BEACON** is a research project building a high-quality, unified cybersecurity Named Entity Recognition (NER) benchmark. It harmonizes four heterogeneous cybersecurity corpora under the [STIX 2.1](https://oasis-open.github.io/cti-documentation/stix/intro) ontology, systematically audits and repairs dataset quality issues, and benchmarks BERT-CRF transformer variants under rigorous, reproducible conditions.

> **Citing prior work**: BEACON builds on and acknowledges [APTNER](https://github.com/), [CyNER](https://github.com/), [DNRTI](https://github.com/), and [Attacker](https://github.com/) datasets. Our contribution is the harmonization pipeline, quality audit + repair, unified benchmark splits, and model training framework. All source datasets retain their original licenses.

---

## Table of Contents

1. [Why BEACON?](#why-beacon)
2. [Dataset](#dataset)
3. [Benchmark Results](#benchmark-results)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Repository Structure](#repository-structure)
7. [Roadmap](#roadmap)
8. [Citation](#citation)
9. [License](#license)

---

## Why BEACON?

Cybersecurity NER is fragmented: every corpus uses its own incompatible entity schema, preventing cross-dataset comparison, transfer learning, and honest model evaluation. BEACON addresses this by:

- **Harmonizing** four publicly available cybersecurity corpora onto a single STIX 2.1 taxonomy via a documented, reproducible mapping pipeline.
- **Auditing and repairing** quality issues — BIO violations, duplicate leakage, truncated documents, class imbalance, and cross-source label conflicts.
- **Benchmarking** BERT-CRF transformer variants (general and domain-adapted) under identical, reproducible conditions with source-stratified splits.
- **Publishing** the cleaned corpus, splits, trained models, and all code — enabling the community to build directly on this foundation.

---

## Dataset

### Sources (v1 — harmonized, pre-cleaning)

| Source | Sentences | Tokens | Original Schema | Description |
|--------|----------:|-------:|-----------------|-------------|
| APTNER | 10,042 | 260,290 | 11 APT-centric types | APT campaign reports |
| CyNER | 4,372 | 106,991 | 5 general cyber types | General cybersecurity text |
| DNRTI | 6,582 | 175,679 | 11 CTI types | Dark-net threat intelligence |
| Attacker | 2,481 | 66,962 | 8 attacker-centric types | Attacker-perspective reports |
| **Total** | **23,477** | **609,922** | **21 STIX 2.1 types** | |

### STIX 2.1 Entity Types

| STIX Type | Entity Count | Primary Sources |
|-----------|-------------:|-----------------|
| Identity | 10,577 | APTNER, DNRTI, CyNER, Attacker |
| Threat-Actor | 9,408 | APTNER, DNRTI |
| Malware | 7,700 | APTNER, CyNER, DNRTI, Attacker |
| Tool | 7,045 | APTNER, DNRTI, Attacker |
| Location | 5,740 | APTNER, DNRTI |
| Attack-Pattern | 5,293 | APTNER, DNRTI, Attacker |
| Vulnerability | 2,582 | CyNER, DNRTI, Attacker |
| File | 2,169 | APTNER |
| Indicator | 1,622 | CyNER |
| Software | 1,587 | CyNER |
| Malware-Analysis | 1,161 | DNRTI |
| Infrastructure | 408 | Attacker |
| Domain-Name | 373 | APTNER |
| Observed-Data | 333 | Attacker |
| Network-Traffic | 330 | Attacker |
| Course-of-Action | 266 | Attacker |
| Campaign | 226 | Attacker |
| IPv4-Addr | 215 | APTNER |
| Intrusion-Set | 138 | Attacker |
| URL | 92 | CyNER |
| Email-Addr | 35 | CyNER |

### Harmonization

Each source's original labels are mapped to STIX 2.1 types via a documented pipeline in [`notebooks/0_data_merging.ipynb`](notebooks/0_data_merging.ipynb). The harmonized label is stored in the `STIX_Tag` column; the original source label is preserved in `Tag`.

| Source | Key Mappings |
|--------|-------------|
| APTNER | `APT→Threat-Actor`, `MAL→Malware`, `TOOL→Tool`, `FILE→File`, `IP→IPv4-Addr`, `ACT→Attack-Pattern`, `TIME/ENCR→O` |
| CyNER | `Malware→Malware`, `System→Software`, `Organization→Identity`, `Indicator→Indicator`, `Vulnerability→Vulnerability` |
| DNRTI | `HackOrg→Threat-Actor`, `SamFile→Malware`, `Features→Malware-Analysis`, `Exp→Vulnerability`, `Way/OffAct→Attack-Pattern`, `Time/Purp→O` |
| Attacker | `VICTIM_IDENTITY/GENERAL_IDENTITY→Identity`, `ATTACK_TOOL→Tool`, `MALWARE→Malware`, `CAMPAIGN→Campaign`, `IMPACT/MOTIVATION→O` |

### Known Quality Issues (being addressed — see Roadmap)

| Issue | Count | Status |
|-------|------:|--------|
| BIO sequence violations | 430 | 🔄 Phase 1 |
| Null token rows (CSV NA artefact) | 82 | 🔄 Phase 1 |
| Duplicate sentence groups crossing train/test boundary | 2,056 groups / 6,298 sentences | 🔄 Phase 1 |
| Sentences with conflicting cross-source annotations | 1,985 groups / 6,118 sentences | 📋 Phase 2 |
| Documents >256 tokens (silently truncated) | 20 sentences; max = 5,862 tokens | 🔄 Phase 1 |
| STIX classes with <300 training instances | 10 types | 📋 Phase 3 |

---

## Benchmark Results

> Strict entity-level F1 (seqeval). All models use BERT-CRF architecture, AMP bfloat16, 256 max tokens, source-stratified splits (seed 42). Dataset: v1 harmonized corpus (pre-cleaning).

| Model | Test F1 | APTNER | CyNER | DNRTI | Attacker |
|-------|--------:|-------:|------:|------:|---------:|
| `roberta-base` | — | — | — | — | — |
| `cisco-ai/SecureBERT2.0-base` | — | — | — | — | — |
| `microsoft/deberta-v3-base` | — | — | — | — | — |
| `answerdotai/ModernBERT-base` | — | — | — | — | — |
| `ehsanaghaei/SecureBERT` | — | — | — | — | — |
| `markusbayer/CySecBERT` | — | — | — | — | — |

*Benchmark sweep in progress. See [Issues](../../issues) for current training status.*

---

## Installation

Requires Python 3.10+ and a CUDA-capable GPU (tested on RTX 3080 16 GB).

```bash
git clone https://github.com/neostrange/BEACON.git
cd BEACON

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Training a Model

```bash
# Reproduce the roberta-base baseline
python scripts/train.py \
  --dataset_path dataset/beacon_stix_v1.csv \
  --model_type roberta-base \
  --output_dir outputs \
  --split_file outputs/splits.json \
  --batch_size 16 --use_amp --seed 42 \
  --epochs 50 --early_stopping_patience 8

# Domain-adapted cybersecurity model (SecureBERT)
python scripts/train.py \
  --model_type ehsanaghaei/SecureBERT \
  --dataset_path dataset/beacon_stix_v1.csv \
  --output_dir outputs \
  --split_file outputs/splits.json \
  --batch_size 16 --use_amp --seed 42

# Large model with gradient checkpointing (DeBERTa-v3-large)
python scripts/train.py \
  --model_type microsoft/deberta-v3-large \
  --dataset_path dataset/beacon_stix_v1.csv \
  --output_dir outputs \
  --split_file outputs/splits.json \
  --batch_size 4 --gradient_accumulation_steps 4 \
  --gradient_checkpointing --use_amp --seed 42

# Full benchmark sweep (6 models sequentially)
bash scripts/run_local.sh
```

### Evaluation Only (from saved checkpoint)

```bash
python scripts/train.py \
  --model_type roberta-base \
  --dataset_path dataset/beacon_stix_v1.csv \
  --output_dir outputs \
  --split_file outputs/splits.json \
  --eval_only
```

Results are written to `outputs/<model_name>_unified/final_evaluation_results.json`.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `roberta-base` | Any HuggingFace model ID |
| `--batch_size` | `16` | Per-device batch size |
| `--gradient_accumulation_steps` | `1` | Effective batch = batch × accum |
| `--learning_rate` | `3e-5` | Encoder learning rate |
| `--lr_crf_fc` | `8e-5` | CRF / linear head learning rate |
| `--epochs` | `50` | Maximum training epochs |
| `--early_stopping_patience` | `8` | Epochs without val F1 improvement |
| `--use_amp` | `False` | Enable bfloat16 mixed-precision |
| `--gradient_checkpointing` | `False` | Reduce VRAM for large encoders |
| `--max_seq_length` | `256` | Token sequence length |
| `--split_file` | — | Path to pre-computed split JSON (reuse for fair comparison) |
| `--eval_only` | `False` | Skip training; evaluate saved checkpoint |
| `--seed` | `42` | Reproducibility seed |

---

## Repository Structure

```
BEACON/
├── dataset/
│   ├── beacon_stix_v1.csv     # Unified harmonized corpus v1 (23,477 sentences)
│   └── versions/                      # Cleaned/versioned dataset releases
├── notebooks/
│   ├── 0_data_merging.ipynb           # Schema harmonization pipeline
│   ├── 1_anlyse_source_results.ipynb  # Per-source analysis
│   ├── 2_analyse_beacon_results.ipynb
│   ├── 3_beacon_training.ipynb
│   └── 4_beacon_testing.ipynb
├── scripts/
│   ├── train.py                       # BERT-CRF training + evaluation (main)
│   ├── run_local.sh                   # Full benchmark sweep runner
│   ├── models_cyber.py                # Legacy training code (reference only)
│   └── models_source.py               # Legacy per-source training (reference only)
├── outputs/
│   └── splits.json                    # Reproducible source-stratified split
├── requirements.txt
└── README.md
```

---

## Roadmap

Active development is tracked via [GitHub Issues](../../issues). The plan targets publishable quality (>86% entity F1 on the cleaned benchmark):

| Phase | Status | Goal |
|-------|--------|------|
| **0 – Baseline Sweep** | 🔄 In Progress | Train 6 BERT-CRF variants on v1 corpus; establish baselines |
| **Phase 1 – Dataset Cleanup** | 📋 Planned | BIO repair, null-token removal, document segmentation, duplicate-grouped splits |
| **Phase 2 – Adjudication** | 📋 Planned | Human review of 1,985 cross-source annotation conflicts |
| **Phase 3 – Rare Class Augmentation** | 📋 Planned | Back-translation / paraphrase for 10 STIX types with <300 training instances |
| **Phase 4 – Publication Run** | 📋 Planned | DeBERTa-v3-large on cleaned corpus; ablation study; paper submission |

---

## Citation

If you use BEACON in your research, please cite (preprint forthcoming):

```bibtex
@misc{beacon2026,
  title  = {BEACON: Benchmark for Entity Recognition Across Cybersecurity Sources with Unified ONtology},
  author = {Ech-Chammakhy, Yasir and others},
  year   = {2026},
  url    = {https://github.com/neostrange/BEACON}
}
```

Please also cite the original datasets:

```bibtex
@dataset{aptner,   title={APTNER},           ... }
@dataset{cyner,    title={CyNER},            ... }
@dataset{dnrti,    title={DNRTI},            ... }
@dataset{attacker, title={Attacker NER},     ... }
```

---

## License

MIT — see [LICENSE](LICENSE).
