# BEACON — Vision, Roadmap & Research Direction

> Last updated: May 2026. This document is the authoritative planning reference for the project.
> Keep it up to date as phases complete and new directions are decided.

---

## Vision

**Make BEACON the community-trusted NER benchmark for cybersecurity** — the single number a paper reports when claiming SOTA on cybersecurity entity extraction.

The field currently has many individual datasets (CyNER, APTNER, DNRTI, AttackER) but no shared, clean, fairly-evaluated benchmark. BEACON's goal is to fix that: a harmonized, reproducible, continuously-improving corpus and leaderboard that the community can build on — the way CoNLL-2003 anchored general NER for a decade.

**Measurable target**: >86% entity-level F1 on a cleaned, deduplicated v2 corpus with a published leaderboard.

BEACON directly extends the [CyberNER paper (arXiv:2510.26499)](https://arxiv.org/abs/2510.26499) accepted at IEEE TrustCom 2025. That paper proved harmonization works (+30% relative F1 over naive concatenation). BEACON's job is to turn that into something rigorous, reproducible, and publication-ready.

---

## Current State (May 2026)

| Item | Status |
|------|--------|
| Harmonized v1 corpus | ✅ Released (`dataset/beacon_stix_v1.csv`, 23,477 sentences / 609K tokens) |
| Reproducible splits | ✅ `outputs/splits.json` (source-stratified, seed 42) |
| `roberta-base` baseline | ✅ **69.6% entity F1** (43 epochs, early stopping) |
| Remaining 5 model baselines | 🔄 In progress (`bash scripts/run_local.sh`) |
| Dataset quality fixes (v2) | 📋 Planned |
| Conflict adjudication | 📋 Planned |
| Rare class augmentation | 📋 Planned |
| Publication run | 📋 Planned |

The current 69.6% F1 on v1 is the pre-cleaning baseline. It is expected to rise substantially once duplicate leakage is fixed and the corpus is cleaned (Phases 1–2).

---

## Roadmap

### Phase 0 — Baseline Sweep *(in progress)*

**Goal**: Fill the results table with all 6 BERT-CRF variants on the v1 corpus.

**Why first**: Establishes pre-cleaning baselines for every model. Determines whether domain-adapted models (SecureBERT, CySecBERT) outperform general ones on dirty data. `roberta-base` at 69.6% F1 is the anchor.

**Models in sweep** (`scripts/run_local.sh`):

| Model | Type | Batch | Status |
|-------|------|-------|--------|
| `roberta-base` | General | 16 | ✅ 69.6% F1 |
| `cisco-ai/SecureBERT2.0-base` | Cyber domain | 16 | 📋 |
| `microsoft/deberta-v3-base` | General (stronger) | 10×2 | 📋 |
| `answerdotai/ModernBERT-base` | General (modern) | 16 | 📋 |
| `ehsanaghaei/SecureBERT` | Cyber domain | 16 | 📋 |
| `markusbayer/CySecBERT` | Cyber domain | 16 | 📋 |

**Done when**: All 6 rows in the README results table are filled.

---

### Phase 1 — Dataset Cleanup → v2 corpus

**Goal**: Fix the 4 known quality issues in `beacon_stix_v1.csv` to produce `beacon_stix_v2.csv`.

| Issue | Count | Fix | Notebook |
|-------|------:|-----|----------|
| BIO sequence violations | 430 | Auto-repair (re-span boundaries) | `0_data_merging.ipynb` |
| Null token rows (CSV NA artefact) | 82 | Drop rows | `0_data_merging.ipynb` |
| Documents >256 tokens (silently truncated) | 20 sentences; max 5,862 tokens | Sliding-window segmentation with overlap | new |
| Duplicate sentence groups crossing train/test | 2,056 groups / 6,298 sentences | Re-split with dedup-aware grouping (group → split, not sentence → split) | new |

**The duplicate leakage is the most critical fix.** It inflates test scores artificially and makes current numbers unreliable for publication.

**Done when**: `dataset/beacon_stix_v2.csv` + `outputs/splits_v2.json` (dedup-grouped) published.

---

### Phase 2 — Conflict Adjudication (LLM-assisted)

**Goal**: Resolve 1,985 sentence groups where the same text span is annotated differently across source datasets (e.g., "Cobalt Strike" → `Tool` in APTNER, `Malware` in DNRTI).

**Originally planned as manual human review. Now: LLM-assisted with human spot-check.**

#### LLM adjudication pipeline (`notebooks/5_llm_adjudication.ipynb`)

```
System: You are a cybersecurity annotation expert. Given a sentence and a
        disputed entity span, assign the single most appropriate STIX 2.1
        type. Respond in JSON: {"label": "...", "confidence": 0-1, "reasoning": "..."}

User:   Sentence: "The group deployed Cobalt Strike beacons on compromised hosts."
        Disputed span: "Cobalt Strike"
        Source A says: Tool   (APTNER)
        Source B says: Malware  (DNRTI)
        STIX definitions: Tool = legitimate software repurposed for attack;
                          Malware = software designed to damage/compromise systems.
```

**Workflow**:
1. Extract conflict groups (from notebook 0 analysis — already identified)
2. Batch-prompt LLM (GPT-4o / Claude Sonnet / local Llama-3.1-70B)
3. Auto-accept decisions with confidence ≥ 0.85
4. Human spot-check the remaining ~15% (low-confidence or reasoning gaps)
5. Produce `adjudicated_conflicts.json` → merge into v2 corpus

**Expected outcome**: ~1,785 auto-resolved + ~200 human-reviewed. Turns weeks of manual work into ~1 day.

**Done when**: `adjudicated_conflicts.json` published + merged into `beacon_stix_v2.csv`.

---

### Phase 3 — Rare Class Augmentation (LLM-assisted)

**Goal**: Bring the 10 STIX types with <300 training instances up to a minimum viable count (~300–500 each) using LLM-generated synthetic data.

| STIX Type | Current Count | Target |
|-----------|-------------:|-------:|
| Email-Addr | 35 | 300 |
| URL | 92 | 300 |
| IPv4-Addr | 215 | 300 |
| Intrusion-Set | 138 | 300 |
| Campaign | 226 | 300 |
| Course-of-Action | 266 | 300 |
| Network-Traffic | 330 | 400 |
| Observed-Data | 333 | 400 |
| Domain-Name | 373 | 400 |
| Infrastructure | 408 | 400 |

#### Augmentation pipeline (`notebooks/6_llm_augmentation.ipynb`)

Prompt the LLM to generate new CTI sentences containing a specific entity type, using existing annotated examples as few-shot demonstrations. Output directly in BIO CoNLL format. Validate with:
- BIO consistency checker (hard filter)
- Semantic plausibility check (LLM self-critique)
- Embedding similarity to real examples (soft filter, remove outliers)

**Done when**: `dataset/synthetic_rare_v1.conll` published + integrated into v2 training splits.

---

### Phase 4 — Publication Run

**Goal**: Train the strongest models on the cleaned + augmented v2 corpus; run ablation study; submit the BEACON paper.

**Ablation design**:

| Condition | Corpus | Split | Purpose |
|-----------|--------|-------|---------|
| A | v1 | random | Reproduces CyberNER paper |
| B | v1 | dedup-grouped | Honest baseline |
| C | v2 (cleaned) | dedup-grouped | Effect of cleanup |
| D | v2 + augmentation | dedup-grouped | Effect of rare class boost |
| E | v2 + adjudicated | dedup-grouped | Effect of conflict resolution |
| F | v2 + augmentation + adjudicated | dedup-grouped | Full BEACON |

**Models**: DeBERTa-v3-large + SecureBERT2.0-base (best from Phase 0).

**Done when**: Full results table published + paper submitted.

---

## Longer-term Directions

These are not in the immediate roadmap but are well-motivated and discussed below.

### A. Fusing Additional Datasets

Several public cybersecurity NER datasets are not yet in BEACON:

| Dataset | Entities | Why Add |
|---------|----------|---------|
| **STUCCO** (Bridges et al. 2014) | General cyber | Already cited in CyberNER paper; straightforward fusion |
| **MalwareTextDB** (Lim et al. 2017) | Malware attributes, capabilities, actions | Boosts `Malware` sub-type coverage significantly |
| **SemEval-2018 Task 8** | `Action`, `Entity`, `Modifier` | Malware report extraction; spans map to several STIX types |
| **CASIE** (Satyapanich et al. 2020) | Cybersecurity events + arguments | Adds `Attack-Pattern` and `Course-of-Action` coverage |
| **ThreatZoom / ThreatKG** | CVE, CWE, product names | Strong `Vulnerability` + `Software` coverage — two weak classes |
| **Dark web forum corpora** | Threat actors, malware, tools | Covers underground marketplace language; complements formal CTI |

Adding STUCCO + MalwareTextDB alone could add ~50K tokens and meaningfully boost the 10 rare STIX classes without requiring new annotation.

### B. Silver-Label Annotation at Scale

Use the best Phase 0 model to auto-annotate large unlabeled CTI corpora — threat reports, CVE/NVD descriptions, vendor security blogs — filtered by confidence threshold. This is how most large NLP corpora scale beyond what humans can annotate manually.

```
Unlabeled CTI text
  → BERT-CRF (best Phase 0 model)
  → confidence ≥ 0.90 → silver-label training data
  → confidence 0.75–0.90 → human review queue
  → confidence < 0.75 → discard
```

### C. Cross-lingual Expansion

A significant portion of real CTI originates in **Chinese, Russian, Farsi, and Korean** (primary APT actor languages). Translating a subset and annotating it — or using cross-lingual transfer from mBERT / XLM-R — would make BEACON genuinely useful for global threat intelligence and is a distinct publication contribution.

### D. Relation Extraction Layer

Entities in BEACON currently exist in isolation. Adding **STIX Relationship Objects (SROs)** as span-pair annotations — e.g., `[Cobalt Strike]` —[used-by]→ `[APT29]` — would turn BEACON from a NER benchmark into a knowledge graph extraction benchmark. This is a substantially higher-impact resource for the community and a natural extension given the STIX framework already defines these relationships.

Relevant STIX SROs to annotate:
- `uses` (Threat-Actor / Intrusion-Set → Malware / Tool)
- `targets` (Threat-Actor → Identity / Location / Vulnerability)
- `indicates` (Indicator → Malware / Threat-Actor)
- `mitigates` (Course-of-Action → Attack-Pattern / Malware)

### E. Document-level Annotation

BEACON is currently sentence-level. Many entities only resolve in document context (coreference: "the group" referring to "APT29" two sentences earlier). Shifting to document-level windows with coreference chains improves both annotation quality and model utility for real CTI pipelines.

---

## Annotation Scheme Analysis: STIX 2.1

### Why STIX 2.1 was chosen
- Industry standard — CTI platforms (MISP, OpenCTI, Splunk ES) already consume it
- Well-defined semantics that resolve cross-dataset ambiguities
- Structured vocabulary covering both high-level SDOs (Threat-Actor, Malware) and observables (IPv4-Addr, File)
- Widely adopted and maintained by OASIS

### Known limitations

| Problem | Example | Impact |
|---------|---------|--------|
| **Too coarse in places** | Ransomware, trojans, spyware all → `Malware` | Loses attribution-relevant distinctions |
| **Missing modern types** | No `Container-Image`, `Cloud-Resource`, `Cryptocurrency-Address`, `Registry-Key` as first-class NER entities | Cloud-era and crypto-ransomware attacks poorly covered |
| **Threat-Actor conflates groups and individuals** | Nation-state APT and lone hacktivist both → `Threat-Actor` | Hurts attribution research |
| **Identity is a catch-all** | Victims, organizations, persons, governments all → `Identity` | Too broad for most downstream tasks |
| **No severity/confidence at span level** | STIX has it as a relationship property only | Useful for triage pipelines |

### Recommended evolution: two-level annotation

Rather than abandoning STIX 2.1 (which would break interoperability), introduce **optional sub-type annotations** within the coarse STIX types:

```
Level 1 (current, required):   B-Malware
Level 2 (new, optional):       B-Malware/Ransomware
```

Sub-types to define:

| STIX Type | Proposed Sub-types |
|-----------|-------------------|
| `Malware` | `Ransomware`, `Trojan`, `Worm`, `Spyware`, `Rootkit`, `Backdoor`, `Dropper` |
| `Threat-Actor` | `Nation-State`, `Criminal-Group`, `Hacktivist`, `Insider` |
| `Identity` | `Organization`, `Person`, `Government`, `Sector` |
| `Attack-Pattern` | maps to MITRE ATT&CK tactic (optional) |

This is how CoNLL evolved from 4 flat types to fine-grained OntoNotes 18 types while remaining backward compatible. Level-1-only evaluation preserves comparability with the existing CyberNER paper; Level-2 evaluation is a new, harder task.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Oct 2025 | Chose STIX 2.1 as target taxonomy | Industry standard; resolves cross-dataset ambiguity; enables CTI tool interoperability |
| Oct 2025 | Source-stratified splits (not random) | Prevents data leakage across sources; more honest generalization measure |
| May 2026 | LLM-assisted adjudication (Phase 2) | Manual review of 1,985 groups is impractical; LLM + spot-check achieves same quality in a fraction of the time |
| May 2026 | Keep STIX 2.1, add sub-types (Level 2) | Breaking STIX compatibility loses interoperability value; sub-types add granularity without breaking existing benchmarks |
