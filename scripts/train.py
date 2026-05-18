"""
BEACON – BERT-CRF Training Script
BEACON: Benchmark for Entity recognition Across Cybersecurity sources with unified ONtology
  - AMP mixed-precision (bfloat16) for RTX 3080
  - Source-stratified reproducible train/dev/test split (saved to JSON)
  - seqeval strict entity-level F1 as primary metric (plus legacy token F1)
  - Generalized encoder: auto-detects token_type_ids via config
  - Gradient checkpointing option for large encoders
  - Any HuggingFace model ID accepted without special-casing
  - Configurable dropout
  - Correct CLS/SEP special tokens per tokenizer family
"""

import os
import json
import time
import argparse
import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SOURCES_TO_EVALUATE = ['APTNER', 'CyNER', 'Attacker', 'DNRTI']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, args):
        self.model_type              = args.model_type
        self.max_seq_length          = args.max_seq_length
        self.batch_size              = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.total_train_epochs      = args.epochs
        self.output_dir              = args.output_dir
        self.learning_rate           = args.learning_rate
        self.lr_crf_fc               = args.lr_crf_fc
        self.weight_decay_crf_fc     = args.weight_decay_crf_fc
        self.weight_decay_finetune   = args.weight_decay_finetune
        self.warmup_proportion       = args.warmup_proportion
        self.early_stopping_patience = args.early_stopping_patience
        self.checkpoint_freq         = args.checkpoint_freq
        self.device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed                    = args.seed
        self.max_grad_norm           = args.max_grad_norm
        self.test_size               = args.test_size
        self.val_size                = args.val_size
        self.num_workers             = args.num_workers
        self.dataset_path            = args.dataset_path
        self.split_file              = args.split_file
        self.use_amp                 = args.use_amp and torch.cuda.is_available()
        self.gradient_checkpointing  = args.gradient_checkpointing
        self.dropout                 = args.dropout

        os.makedirs(self.output_dir, exist_ok=True)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InputExample:
    def __init__(self, guid, words, labels, source):
        self.guid   = guid
        self.words  = words
        self.labels = labels
        self.source = source


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids    = input_ids
        self.input_mask   = input_mask
        self.segment_ids  = segment_ids
        self.predict_mask = predict_mask
        self.label_ids    = label_ids


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_unified_dataset(df):
    # Group by (Source, Sentence_ID) because Sentence_ID is source-local and
    # different sources reuse the same integer IDs.
    grouped  = df.groupby(['Source', 'Sentence_ID'])
    examples = []
    print(f"Processing {len(grouped)} unique sentences …")
    for (source, sentence_id), group in tqdm(grouped, desc="Preparing Examples"):
        if group.empty:
            continue
        words  = group['Word'].tolist()
        labels = group['STIX_Tag'].tolist()
        examples.append(InputExample(
            guid=f"{source}_{sentence_id}",
            words=words,
            labels=labels,
            source=source,
        ))
    print(f"Created {len(examples)} sentence examples.")
    return examples


def source_stratified_split(examples, test_size, val_size, seed, split_file=None):
    """
    Per-source stratified split.  Saves/reuses split IDs from split_file so
    every model benchmark uses exactly the same sentences.
    """
    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            saved = json.load(f)
        g = {ex.guid: ex for ex in examples}
        train_ex = [g[id_] for id_ in saved['train'] if id_ in g]
        dev_ex   = [g[id_] for id_ in saved['dev']   if id_ in g]
        test_ex  = [g[id_] for id_ in saved['test']  if id_ in g]
        print(f"Loaded fixed split from {split_file}: "
              f"Train={len(train_ex)}, Val={len(dev_ex)}, Test={len(test_ex)}")
        return train_ex, dev_ex, test_ex

    source_groups = defaultdict(list)
    for ex in examples:
        source_groups[ex.source].append(ex)

    train_all, dev_all, test_all = [], [], []
    rng = np.random.RandomState(seed)

    print("Creating source-stratified split:")
    for source, exs in sorted(source_groups.items()):
        n      = len(exs)
        perm   = rng.permutation(n)
        exs    = [exs[i] for i in perm]
        n_test = max(1, int(round(n * test_size)))
        n_val  = max(1, int(round(n * val_size)))
        n_train = max(1, n - n_test - n_val)
        # Correct rounding drift
        while n_train + n_test + n_val > n:
            n_val -= 1
        test_all.extend(exs[:n_test])
        dev_all.extend(exs[n_test:n_test + n_val])
        train_all.extend(exs[n_test + n_val:])
        print(f"  {source}: total={n}, train={n_train}, val={n_val}, test={n_test}")

    if split_file:
        os.makedirs(os.path.dirname(os.path.abspath(split_file)), exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump({
                'train': [ex.guid for ex in train_all],
                'dev':   [ex.guid for ex in dev_all],
                'test':  [ex.guid for ex in test_all],
            }, f, indent=2)
        print(f"Saved split IDs to {split_file}")

    return train_all, dev_all, test_all


# ---------------------------------------------------------------------------
# Feature conversion
# ---------------------------------------------------------------------------

def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'

    # Use the tokenizer's actual special tokens so RoBERTa/ModernBERT get <s>/<\/s>
    cls_token = tokenizer.cls_token or '[CLS]'
    sep_token = tokenizer.sep_token or '[SEP]'

    tokens       = [cls_token]
    label_ids    = [label_map.get('[CLS]', label_map['O'])]
    predict_mask = [0]

    for i, word in enumerate(example.words):
        sub = tokenizer.tokenize(str(word))
        if not sub:
            sub = [tokenizer.unk_token or '[UNK]']
        tokens.extend(sub)
        for j in range(len(sub)):
            if j == 0:
                label_ids.append(label_map.get(example.labels[i], label_map['O']))
                predict_mask.append(1)
            else:
                label_ids.append(label_map.get(add_label, label_map['O']))
                predict_mask.append(0)

    # Truncate to leave room for [SEP]
    if len(tokens) > max_seq_length - 1:
        tokens       = tokens[:max_seq_length - 1]
        label_ids    = label_ids[:max_seq_length - 1]
        predict_mask = predict_mask[:max_seq_length - 1]

    tokens.append(sep_token)
    label_ids.append(label_map.get('[SEP]', label_map['O']))
    predict_mask.append(0)

    input_ids   = tokenizer.convert_tokens_to_ids(tokens)
    input_mask  = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids) == len(predict_mask)
    return InputFeatures(input_ids, input_mask, segment_ids, predict_mask, label_ids)


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.features = []
        print(f"Converting {len(examples)} examples to features …")
        for ex in tqdm(examples, desc="Creating Features"):
            try:
                self.features.append(example2feature(ex, tokenizer, label_map, max_seq_length))
            except Exception as e:
                print(f"Skipping example {ex.guid}: {e}")
        print(f"Created {len(self.features)} features.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return f.input_ids, f.input_mask, f.segment_ids, f.predict_mask, f.label_ids

    @staticmethod
    def pad(batch):
        maxlen = max(len(s[0]) for s in batch)
        pad = lambda x, v: [s[x] + [v] * (maxlen - len(s[x])) for s in batch]
        return (
            torch.LongTensor(pad(0, 0)),
            torch.LongTensor(pad(1, 0)),
            torch.LongTensor(pad(2, 0)),
            torch.BoolTensor(pad(3, 0)),
            torch.LongTensor(pad(4, 0)),
        )


# ---------------------------------------------------------------------------
# CRF helpers
# ---------------------------------------------------------------------------

def log_sum_exp_batch(log_tensor, axis=-1):
    if log_tensor.nelement() == 0:
        out_shape = list(log_tensor.shape)
        if axis is not None:
            del out_shape[axis]
        return torch.full(out_shape, -float('inf'),
                          device=log_tensor.device, dtype=log_tensor.dtype)
    max_score = torch.max(log_tensor, axis, keepdim=True)[0]
    max_score[torch.isneginf(max_score)] = 0.0
    return (torch.exp(log_tensor - max_score).sum(axis, keepdim=True).log()
            + max_score).squeeze(axis)


# ---------------------------------------------------------------------------
# Token-level metrics (kept for backward compatibility / comparison)
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, label_map, idx2label=None):
    if idx2label is None:
        idx2label = {v: k for k, v in label_map.items()}

    ignore_ids = {label_map[l] for l in ('O', 'X', '[CLS]', '[SEP]') if l in label_map}

    true_valid = np.isin(y_true, list(ignore_ids), invert=True)
    pred_valid = np.isin(y_pred, list(ignore_ids), invert=True)

    num_gold     = np.sum(true_valid)
    num_proposed = np.sum(pred_valid)
    num_correct  = np.sum(np.logical_and(y_true == y_pred, true_valid))

    p   = num_correct / num_proposed if num_proposed > 0 else 0.0
    r   = num_correct / num_gold     if num_gold     > 0 else 0.0
    f1  = 2 * p * r / (p + r)       if (p + r)      > 0 else 0.0

    class_metrics = {}
    for cls_id in sorted(idx2label.keys()):
        if cls_id in ignore_ids:
            continue
        tp = np.sum(np.logical_and(y_true == cls_id, y_pred == cls_id))
        fp = np.sum(np.logical_and(y_true != cls_id, y_pred == cls_id))
        fn = np.sum(np.logical_and(y_true == cls_id, y_pred != cls_id))
        cp  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        class_metrics[cls_id] = {
            'label': idx2label[cls_id],
            'precision': cp, 'recall': cr, 'f1': cf1, 'support': int(tp + fn),
        }
    return p, r, f1, class_metrics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BERT_CRF_NER(nn.Module):
    def __init__(self, model_name, start_label_id, stop_label_id,
                 num_labels, device, dropout=0.2, gradient_checkpointing=False):
        super().__init__()
        self.num_labels      = num_labels
        self.start_label_id  = start_label_id
        self.stop_label_id   = stop_label_id
        self.device          = device

        print(f"Loading encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")

        # Auto-detect token_type_ids: only use them when type_vocab_size >= 2
        # (BERT=2, RoBERTa=1, ModernBERT/DeBERTa-v3=0)
        self.use_token_type_ids = getattr(self.encoder.config, 'type_vocab_size', 0) >= 2
        print(f"use_token_type_ids={self.use_token_type_ids} "
              f"(type_vocab_size={getattr(self.encoder.config,'type_vocab_size','n/a')})")

        self.hidden_size   = self.encoder.config.hidden_size
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden2label  = nn.Linear(self.hidden_size, num_labels)

        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.transitions.data[start_label_id, :] = -10000.0
        self.transitions.data[:, stop_label_id]  = -10000.0

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _get_encoder_features(self, input_ids, segment_ids, input_mask):
        kwargs = dict(input_ids=input_ids, attention_mask=input_mask)
        if self.use_token_type_ids:
            kwargs['token_type_ids'] = segment_ids
        out = self.encoder(**kwargs)
        seq = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        seq = self.dropout_layer(seq)
        # Cast back to fp32 before CRF to avoid numerical issues under bfloat16
        return self.hidden2label(seq).float()

    # ---------- CRF: forward algorithm ----------
    def _forward_alg(self, feats, mask):
        batch_size, seq_len, _ = feats.shape
        mask_bool = mask.bool()
        log_alpha = torch.full((batch_size, self.num_labels), -10000.0, device=self.device)
        log_alpha[:, self.start_label_id] = 0.0
        trans = self.transitions.unsqueeze(0)
        for t in range(seq_len):
            valid = mask_bool[:, t]
            if not valid.any():
                continue
            emit = feats[:, t]
            scores = log_alpha.unsqueeze(2) + trans + emit.unsqueeze(1)
            next_alpha = log_sum_exp_batch(scores, axis=1)
            log_alpha = torch.where(valid.unsqueeze(1), next_alpha, log_alpha)
        log_alpha += self.transitions[:, self.stop_label_id].unsqueeze(0)
        return log_sum_exp_batch(log_alpha, axis=1)

    # ---------- CRF: gold-sequence score ----------
    def _score_sentence(self, feats, label_ids, mask):
        batch_size, seq_len, _ = feats.shape
        mask_bool = mask.bool()
        score = torch.zeros(batch_size, device=self.device)
        first_labels = label_ids[:, 1] if seq_len > 1 else label_ids[:, 0]
        score += self.transitions[self.start_label_id, first_labels]
        for t in range(seq_len - 1):
            cur  = label_ids[:, t]
            nxt  = label_ids[:, t + 1]
            valid = mask_bool[:, t + 1]
            score += (self.transitions[cur, nxt]
                      + feats[:, t + 1].gather(1, nxt.unsqueeze(1)).squeeze(1)) * valid.float()
        seq_lengths = mask.sum(dim=1).long()
        last_labels = label_ids.gather(1, (seq_lengths - 1).unsqueeze(1)).squeeze(1)
        score += self.transitions[last_labels, self.stop_label_id]
        return score

    # ---------- CRF: Viterbi decode ----------
    def _viterbi_decode(self, feats, mask):
        batch_size, seq_len, _ = feats.shape
        mask_bool = mask.bool()
        log_delta = torch.full((batch_size, self.num_labels), -10000.0, device=self.device)
        log_delta[:, self.start_label_id] = 0.0
        psi   = torch.zeros(batch_size, seq_len, self.num_labels, dtype=torch.long, device=self.device)
        trans = self.transitions.unsqueeze(0)
        for t in range(seq_len):
            valid = mask_bool[:, t].unsqueeze(1)
            if not valid.any():
                continue
            scores     = log_delta.unsqueeze(2) + trans
            max_sc, ix = torch.max(scores, dim=1)
            max_sc    += feats[:, t]
            if t > 0:
                psi[:, t, :] = ix
            log_delta = torch.where(valid, max_sc, log_delta)
        log_delta += self.transitions[:, self.stop_label_id].unsqueeze(0)
        best_scores, last_tags = torch.max(log_delta, dim=1)
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            L = int(mask[b].sum().item())
            if L == 0:
                continue
            best_paths[b, L - 1] = last_tags[b]
            for t in range(L - 2, -1, -1):
                best_paths[b, t] = psi[b, t + 1, best_paths[b, t + 1]]
        return best_scores, best_paths

    # ---------- Loss ----------
    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        mask  = input_mask.float()
        feats = self._get_encoder_features(input_ids, segment_ids, input_mask)
        fwd   = self._forward_alg(feats, mask)
        gold  = self._score_sentence(feats, label_ids, mask)
        return torch.mean(fwd - gold)

    # ---------- Inference ----------
    def forward(self, input_ids, segment_ids, input_mask):
        mask  = input_mask.float()
        feats = self._get_encoder_features(input_ids, segment_ids, input_mask)
        return self._viterbi_decode(feats, mask)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, epoch, dataset_name, label_map, idx2label, device, use_amp=False):
    model.eval()
    # For token-level legacy metric
    all_labels_flat = []
    all_preds_flat  = []
    # For seqeval strict entity metric (per-sentence lists)
    all_true_seqs = []
    all_pred_seqs = []

    special_tags = {'X', '[CLS]', '[SEP]'}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}", leave=False):
            batch  = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            if input_ids.size(0) == 0:
                continue

            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.bfloat16):
                _, predicted = model(input_ids, segment_ids, input_mask)

            for i in range(input_ids.size(0)):
                seq_len   = input_mask[i].sum().item()
                valid_idx = predict_mask[i][:seq_len].nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue

                true_ids = label_ids[i][valid_idx].cpu().numpy()
                pred_ids = predicted[i][valid_idx].cpu().numpy()

                all_labels_flat.extend(true_ids)
                all_preds_flat.extend(pred_ids)

                # Convert to string BIO tags; replace special tags with O
                true_tags = [
                    'O' if idx2label.get(int(l), 'O') in special_tags
                    else idx2label.get(int(l), 'O')
                    for l in true_ids
                ]
                pred_tags = [
                    'O' if idx2label.get(int(p), 'O') in special_tags
                    else idx2label.get(int(p), 'O')
                    for p in pred_ids
                ]
                all_true_seqs.append(true_tags)
                all_pred_seqs.append(pred_tags)

    if not all_labels_flat:
        print(f"Warning: no valid tokens found for {dataset_name}.")
        return 0.0, 0.0, {}, {}

    # --- Token-level (backward compat) ---
    _, _, token_f1, class_metrics = calculate_metrics(
        np.array(all_labels_flat), np.array(all_preds_flat), label_map, idx2label
    )

    # --- Strict entity-level (seqeval) ---
    entity_f1      = 0.0
    seqeval_result = {}
    if all_true_seqs:
        try:
            entity_f1      = seqeval_f1(all_true_seqs, all_pred_seqs, average='micro', zero_division=0)
            seqeval_result = seqeval_report(all_true_seqs, all_pred_seqs, output_dict=True, zero_division=0)
        except Exception as e:
            print(f"seqeval error: {e}")

    print(f"\n--- {dataset_name} (epoch {epoch}) ---")
    print(f"  Strict entity F1  (seqeval): {100.*entity_f1:.2f}%")
    print(f"  Token-level F1    (legacy):  {100.*token_f1:.2f}%")

    if seqeval_result:
        print("  Per-class (seqeval):")
        for cls, m in sorted(seqeval_result.items()):
            if cls in ('micro avg', 'macro avg', 'weighted avg') or not isinstance(m, dict):
                continue
            print(f"    {cls:<30}: P={m.get('precision',0):.4f}  "
                  f"R={m.get('recall',0):.4f}  F1={m.get('f1-score',0):.4f}  "
                  f"S={m.get('support',0)}")
    print("---")

    return entity_f1, token_f1, class_metrics, seqeval_result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---- Load Data ----
    print(f"Loading dataset from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path, keep_default_na=False, encoding='utf-8')
    for col in ('Sentence_ID', 'Word', 'STIX_Tag', 'Source'):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    n_sents = df.groupby(['Source', 'Sentence_ID']).ngroups
    print(f"Loaded {len(df)} rows, {n_sents} sentences.")

    # ---- Label map ----
    base_labels = ['O', 'X', '[CLS]', '[SEP]']
    entity_tags  = sorted(t for t in df['STIX_Tag'].unique() if t not in base_labels)
    all_labels   = base_labels + entity_tags
    label_map    = {l: i for i, l in enumerate(all_labels)}
    idx2label    = {i: l for l, i in label_map.items()}
    print(f"Label map: {len(all_labels)} labels  →  {entity_tags}")
    start_label_id = label_map['[CLS]']
    stop_label_id  = label_map['[SEP]']

    # ---- Prepare examples ----
    all_examples = prepare_unified_dataset(df)
    if not all_examples:
        raise RuntimeError("No examples created from dataset.")

    # ---- Split ----
    train_examples, dev_examples, test_examples = source_stratified_split(
        all_examples, config.test_size, config.val_size, config.seed, config.split_file
    )
    print(f"Split: Train={len(train_examples)}, Val={len(dev_examples)}, Test={len(test_examples)}")

    safe_name    = config.model_type.replace('/', '_')
    model_outdir = os.path.join(config.output_dir, safe_name + "_unified")

    # ---- Eval-only shortcut ----
    if getattr(config, 'eval_only', False):
        hist_path = os.path.join(model_outdir, 'training_history.json')
        if not os.path.exists(hist_path):
            raise FileNotFoundError(f"training_history.json not found at {hist_path}")
        with open(hist_path) as f:
            hist = json.load(f)
        vf1s = hist['valid_entity_f1']
        best_epoch     = int(np.argmax(vf1s)) + 1
        best_entity_f1 = float(max(vf1s))
        print(f"[eval_only] Loaded history: best epoch={best_epoch}, val F1={best_entity_f1:.4f}")
        ckpt_path = os.path.join(model_outdir, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"best_model.pt not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        final_tokenizer = AutoTokenizer.from_pretrained(model_outdir)
        final_model = BERT_CRF_NER(
            model_name=config.model_type,
            start_label_id=start_label_id,
            stop_label_id=stop_label_id,
            num_labels=len(all_labels),
            device=config.device,
            dropout=config.dropout,
        )
        final_model.load_state_dict(ckpt['model_state'])
        final_model.to(config.device)
        final_model.eval()
        pin = config.device.type == 'cuda'
        _run_final_eval(config, final_model, final_tokenizer, test_examples,
                        label_map, idx2label, model_outdir,
                        best_epoch, best_entity_f1, pin)
        return

    # ---- Tokenizer ----
    print(f"Loading tokenizer: {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_type)

    # ---- Datasets ----
    train_ds = NerDataset(train_examples, tokenizer, label_map, config.max_seq_length)
    dev_ds   = NerDataset(dev_examples,   tokenizer, label_map, config.max_seq_length)

    pin = config.device.type == 'cuda'
    train_dl = data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               num_workers=config.num_workers, collate_fn=NerDataset.pad,
                               pin_memory=pin)
    dev_dl   = data.DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False,
                               num_workers=config.num_workers, collate_fn=NerDataset.pad,
                               pin_memory=pin)

    eff_batch = config.batch_size * config.gradient_accumulation_steps
    total_steps = (len(train_dl) // config.gradient_accumulation_steps) * config.total_train_epochs
    print(f"\n{'='*60}")
    print(f"  Model:               {config.model_type}")
    print(f"  Train/Val/Test:      {len(train_examples)}/{len(dev_examples)}/{len(test_examples)}")
    print(f"  Batch / eff batch:   {config.batch_size} / {eff_batch}")
    print(f"  LR encoder/CRF-FC:  {config.learning_rate} / {config.lr_crf_fc}")
    print(f"  AMP:                 {config.use_amp}")
    print(f"  Grad checkpointing:  {config.gradient_checkpointing}")
    print(f"  Device:              {config.device}")
    if config.device.type == 'cuda':
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ---- Model ----
    model = BERT_CRF_NER(
        model_name=config.model_type,
        start_label_id=start_label_id,
        stop_label_id=stop_label_id,
        num_labels=len(all_labels),
        device=config.device,
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    model.to(config.device)

    # ---- Optimizer ----
    no_decay    = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.weight']
    crf_fc_keys = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(k in n for k in crf_fc_keys)],
         'lr': config.learning_rate, 'weight_decay': config.weight_decay_finetune},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(k in n for k in crf_fc_keys)],
         'lr': config.learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters()
                    if n in ('transitions', 'hidden2label.weight')],
         'lr': config.lr_crf_fc, 'weight_decay': config.weight_decay_crf_fc},
        {'params': [p for n, p in model.named_parameters() if n == 'hidden2label.bias'],
         'lr': config.lr_crf_fc, 'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_proportion),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(device='cuda', enabled=config.use_amp)

    # ---- Output dir ----
    safe_name    = config.model_type.replace('/', '_')
    model_outdir = os.path.join(config.output_dir, safe_name + "_unified")
    os.makedirs(model_outdir, exist_ok=True)

    # ---- Training loop ----
    best_entity_f1       = 0.0
    early_stop_counter   = 0
    best_epoch           = 0
    history = {'train_loss': [], 'valid_entity_f1': [], 'valid_token_f1': [],
               'best_valid_entity_f1': 0.0, 'epochs_ran': 0}

    print("***** Starting Training *****")
    for epoch in range(config.total_train_epochs):
        model.train()
        tr_loss    = 0.0
        nb_steps   = 0
        t0         = time.time()
        optimizer.zero_grad()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.total_train_epochs}")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(config.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            with torch.amp.autocast(device_type='cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                nb_steps += 1
                pbar.set_postfix({'loss': f"{tr_loss / nb_steps:.4f}"})

        avg_loss = tr_loss / max(nb_steps, 1)
        history['train_loss'].append(avg_loss)
        history['epochs_ran'] = epoch + 1
        print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}  time={((time.time()-t0)/60):.1f}m")

        if config.device.type == 'cuda':
            alloc = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak VRAM this epoch: {alloc:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # ---- Validation ----
        entity_f1, token_f1, _, _ = evaluate(
            model, dev_dl, epoch + 1, 'Validation',
            label_map, idx2label, config.device, config.use_amp
        )
        history['valid_entity_f1'].append(entity_f1)
        history['valid_token_f1'].append(token_f1)

        # ---- Checkpoint (primary: seqeval entity F1) ----
        if entity_f1 > best_entity_f1:
            print(f"  ✓ New best entity F1: {entity_f1:.4f}  (token F1: {token_f1:.4f})")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'entity_f1': entity_f1,
                'token_f1': token_f1,
                'config': vars(config),
                'label_map': label_map,
                'idx2label': idx2label,
                'model_type': config.model_type,
            }, os.path.join(model_outdir, 'best_model.pt'))
            tokenizer.save_pretrained(model_outdir)
            best_entity_f1 = entity_f1
            history['best_valid_entity_f1'] = best_entity_f1
            early_stop_counter = 0
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            print(f"  No improvement ({entity_f1:.4f} vs best {best_entity_f1:.4f}). "
                  f"Patience: {early_stop_counter}/{config.early_stopping_patience}")

        if (epoch + 1) % config.checkpoint_freq == 0:
            torch.save({'epoch': epoch + 1, 'model_state': model.state_dict()},
                       os.path.join(model_outdir, f'checkpoint_epoch_{epoch+1}.pt'))

        with open(os.path.join(model_outdir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        if early_stop_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    # ===== Final Test Evaluation =====
    print(f"\n***** Best valid entity F1: {best_entity_f1:.4f} at epoch {best_epoch} *****")
    ckpt_path = os.path.join(model_outdir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print("No best_model.pt found – skipping final evaluation.")
        return

    print(f"Loading best checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    final_model = BERT_CRF_NER(
        model_name=config.model_type,
        start_label_id=start_label_id,
        stop_label_id=stop_label_id,
        num_labels=len(all_labels),
        device=config.device,
        dropout=config.dropout,
    )
    final_model.load_state_dict(ckpt['model_state'])
    final_model.to(config.device)
    final_model.eval()
    final_tokenizer = AutoTokenizer.from_pretrained(model_outdir)

    test_ds = NerDataset(test_examples, final_tokenizer, label_map, config.max_seq_length)
    test_dl = data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=NerDataset.pad,
                              pin_memory=pin)

    overall_ent_f1, overall_tok_f1, overall_class, overall_seq_report = evaluate(
        final_model, test_dl, 'Final', 'Test (Overall)',
        label_map, idx2label, config.device, config.use_amp
    )

    final_results = {
        'model_type': config.model_type,
        'best_epoch': best_epoch,
        'best_valid_entity_f1': best_entity_f1,
        'overall_test_entity_f1': overall_ent_f1,
        'overall_test_token_f1': overall_tok_f1,
        'overall_seqeval_report': overall_seq_report,
        'overall_class_metrics': {
            m['label']: {k: v for k, v in m.items() if k != 'label'}
            for m in overall_class.values()
        },
        'source_specific_test_results': {},
    }

    for src in SOURCES_TO_EVALUATE:
        src_exs = [ex for ex in test_examples if ex.source == src]
        if not src_exs:
            final_results['source_specific_test_results'][src] = {'message': 'No examples'}
            continue
        src_ds = NerDataset(src_exs, final_tokenizer, label_map, config.max_seq_length)
        src_dl = data.DataLoader(src_ds, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers, collate_fn=NerDataset.pad,
                                 pin_memory=pin)
        s_ent, s_tok, s_cls, s_seq = evaluate(
            final_model, src_dl, 'Final', f'Test ({src})',
            label_map, idx2label, config.device, config.use_amp
        )
        final_results['source_specific_test_results'][src] = {
            'entity_f1': s_ent, 'token_f1': s_tok,
            'seqeval_report': s_seq,
            'class_metrics': {
                m['label']: {k: v for k, v in m.items() if k != 'label'}
                for m in s_cls.values()
            },
        }

    results_path = os.path.join(model_outdir, 'final_evaluation_results.json')
    with open(results_path, 'w') as f:
        def _ser(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):    return obj.tolist()
            raise TypeError(type(obj))
        json.dump(final_results, f, indent=2, default=_ser)
    print(f"\nFinal results saved to {results_path}")
    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS – {config.model_type}")
    print(f"  Strict entity F1 (seqeval):  {100.*overall_ent_f1:.2f}%")
    print(f"  Token-level F1   (legacy):   {100.*overall_tok_f1:.2f}%")
    print(f"{'='*60}\n")


def _run_final_eval(config, final_model, final_tokenizer, test_examples,
                    label_map, idx2label, model_outdir,
                    best_epoch, best_entity_f1, pin):
    """Run test evaluation and save final_evaluation_results.json."""
    test_ds = NerDataset(test_examples, final_tokenizer, label_map, config.max_seq_length)
    test_dl = data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=NerDataset.pad,
                              pin_memory=pin)

    overall_ent_f1, overall_tok_f1, overall_class, overall_seq_report = evaluate(
        final_model, test_dl, 'Final', 'Test (Overall)',
        label_map, idx2label, config.device, config.use_amp
    )

    final_results = {
        'model_type': config.model_type,
        'best_epoch': best_epoch,
        'best_valid_entity_f1': best_entity_f1,
        'overall_test_entity_f1': overall_ent_f1,
        'overall_test_token_f1': overall_tok_f1,
        'overall_seqeval_report': overall_seq_report,
        'overall_class_metrics': {
            m['label']: {k: v for k, v in m.items() if k != 'label'}
            for m in overall_class.values()
        },
        'source_specific_test_results': {},
    }

    for src in SOURCES_TO_EVALUATE:
        src_exs = [ex for ex in test_examples if ex.source == src]
        if not src_exs:
            final_results['source_specific_test_results'][src] = {'message': 'No examples'}
            continue
        src_ds = NerDataset(src_exs, final_tokenizer, label_map, config.max_seq_length)
        src_dl = data.DataLoader(src_ds, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers, collate_fn=NerDataset.pad,
                                 pin_memory=pin)
        s_ent, s_tok, s_cls, s_seq = evaluate(
            final_model, src_dl, 'Final', f'Test ({src})',
            label_map, idx2label, config.device, config.use_amp
        )
        final_results['source_specific_test_results'][src] = {
            'entity_f1': s_ent, 'token_f1': s_tok,
            'seqeval_report': s_seq,
            'class_metrics': {
                m['label']: {k: v for k, v in m.items() if k != 'label'}
                for m in s_cls.values()
            },
        }

    results_path = os.path.join(model_outdir, 'final_evaluation_results.json')
    with open(results_path, 'w') as f:
        def _ser(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):    return obj.tolist()
            raise TypeError(type(obj))
        json.dump(final_results, f, indent=2, default=_ser)
    print(f"\nFinal results saved to {results_path}")
    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS \u2013 {config.model_type}")
    print(f"  Strict entity F1 (seqeval):  {100.*overall_ent_f1:.2f}%")
    print(f"  Token-level F1   (legacy):   {100.*overall_tok_f1:.2f}%")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BEACON BERT-CRF training')

    # Data
    parser.add_argument('--dataset_path',  required=True)
    parser.add_argument('--split_file',    default=None,
                        help='JSON file to persist/reuse train/dev/test split IDs')
    parser.add_argument('--test_size',     type=float, default=0.15)
    parser.add_argument('--val_size',      type=float, default=0.15)

    # Model
    parser.add_argument('--model_type',    required=True,
                        help='Any HuggingFace model ID, e.g. roberta-base')
    parser.add_argument('--max_seq_length', type=int,   default=256)
    parser.add_argument('--dropout',        type=float, default=0.2)
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save VRAM')

    # Training
    parser.add_argument('--epochs',                     type=int,   default=50)
    parser.add_argument('--batch_size',                 type=int,   default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int,  default=1)
    parser.add_argument('--learning_rate',               type=float, default=3e-5)
    parser.add_argument('--lr_crf_fc',                   type=float, default=8e-5)
    parser.add_argument('--weight_decay_finetune',        type=float, default=1e-5)
    parser.add_argument('--weight_decay_crf_fc',          type=float, default=5e-6)
    parser.add_argument('--warmup_proportion',            type=float, default=0.1)
    parser.add_argument('--max_grad_norm',                type=float, default=1.0)
    parser.add_argument('--early_stopping_patience',      type=int,   default=8)
    parser.add_argument('--checkpoint_freq',              type=int,   default=10)
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable bfloat16 AMP (recommended for RTX 3080)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; load best_model.pt and run final test evaluation only')

    # Env
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir',  default='./outputs/')
    parser.add_argument('--seed',        type=int, default=42)

    args   = parser.parse_args()
    config = Config(args)

    print("\n***** Config *****")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    print("******************\n")

    train(config)


if __name__ == '__main__':
    main()
