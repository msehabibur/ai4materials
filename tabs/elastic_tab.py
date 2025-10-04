# ======================= ONE-SHOT STRONG OFFLINE BASELINE =======================
# Aims for a big bump via: multi-text concat, mean-pooling + multi-sample dropout,
# class weighting + label smoothing, cosine schedule+warmup, balanced sampler,
# and simple TTA at inference. Fully offline; produces submission.csv.

import os, glob, json, random, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# --------- Hard-disable online integrations for Kaggle offline runs ----------
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

# =================== CONFIG ===================
DATA_DIR   = "/kaggle/input/map-charting-student-math-misunderstandings"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

# Prefer larger snapshots if present (still offline)
MODEL_ROOT_HINTS = [
    "/kaggle/input/huggingfacedebertav3variants/khalidalt-DeBERTa-v3-large",
    "/kaggle/input/deberta-v3-large",
    "/kaggle/input/microsoft-deberta-v3-large",
    "/kaggle/input/deberta-v3-base",
    "/kaggle/input/microsoft-deberta-v3-base",
    "/kaggle/input/deberta-v3-small",
    "/kaggle/input/microsoft-deberta-v3-small",
    "/kaggle/input",
]
SUPPORTED_MODEL_TYPES = {"deberta","deberta-v2","deberta-v3"}

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

FP16 = torch.cuda.is_available()
USE_FAST = False           # safer with sentencepiece fallback
MAX_LEN = 256
EPOCHS = 4
TRAIN_BS = 16
VAL_BS = 32
LR = 2e-5                  # slightly lower for large models
WD = 0.02
WARMUP_RATIO = 0.06
GRAD_ACCUM = 2             # effective batch x2
LABEL_SMOOTH = 0.1
N_DROPOUT_PASSES = 5
HEAD_DROPOUT_P = 0.2
TARGET_FOLDS = 5           # we’ll run 1 fold for speed (set >1 to ensemble)

RUN_FOLDS = 1              # change to 5 for CV-ensemble if time/VRAM allows
FOLD_INDEX_TO_RUN = 0      # pick which fold to run when RUN_FOLDS=1

# =================== IO ===================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB)

# =================== LABELS ===================
if {"Category","Misconception"}.issubset(train.columns):
    train["label"] = train["Category"].astype(str).str.strip() + ":" + train["Misconception"].astype(str).str.strip()
else:
    # fallback: take 2nd column as label if contest format changes
    train["label"] = train.iloc[:, 1].astype(str).str.strip()

le = LabelEncoder()
y = le.fit_transform(train["label"].values)
num_labels = len(le.classes_)
print("Num labels:", num_labels)

# =================== TEXT CONCAT ===================
TEXT_CANDS = ["QuestionText","StudentExplanation","student_response","explanation","response",
              "text","answer","content","body","comment"]
present = [c for c in TEXT_CANDS if c in train.columns]

def join_text(df):
    if present:
        cols = present
    else:
        banned = {"Category","Misconception","label","target","row_id","id","example_id"}
        cols = [c for c in df.columns if df[c].dtype==object and c not in banned][:3]
    return df[cols].astype(str).agg(" [SEP] ".join, axis=1)

train["_text"] = join_text(train)
test ["_text"] = join_text(test)
TEXT_COL = "_text"
print(f"Using concatenated text from {len(present) if present else 'fallback'} cols; MAX_LEN={MAX_LEN}")

# =================== MODEL DISCOVERY ===================
def is_supported_model_dir(path: str):
    cfg = os.path.join(path, "config.json")
    if not (os.path.isdir(path) and os.path.exists(cfg)): return (False, None)
    try:
        data = json.load(open(cfg, "r"))
        mtype = str(data.get("model_type","")).lower()
        if mtype in SUPPORTED_MODEL_TYPES:
            tok_files = {"tokenizer.json","spiece.model","vocab.json","merges.txt","tokenizer_config.json"}
            if len(set(os.listdir(path)) & tok_files) > 0:
                return (True, mtype)
    except Exception:
        pass
    return (False, None)

def discover_supported_model_dir(hints):
    # direct hits
    for root in hints:
        if os.path.isdir(root):
            ok, mt = is_supported_model_dir(root)
            if ok: return root, mt
    # shallow children
    for root in hints:
        if os.path.isdir(root):
            for p in glob.glob(os.path.join(root, "*")):
                ok, mt = is_supported_model_dir(p)
                if ok: return p, mt
    # deeper scan
    for p in glob.glob("/kaggle/input/**", recursive=True):
        ok, mt = is_supported_model_dir(p)
        if ok: return p, mt
    return None, None

MODEL_DIR, MODEL_TYPE = discover_supported_model_dir(MODEL_ROOT_HINTS)
if MODEL_DIR is None:
    raise RuntimeError("No DeBERTa snapshot found under /kaggle/input.")
print("Using model:", MODEL_DIR, "| type:", MODEL_TYPE)

# =================== HF Imports ===================
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# =================== Tokenizer/Config ===================
config = AutoConfig.from_pretrained(
    MODEL_DIR,
    num_labels=num_labels,
    id2label={i:l for i,l in enumerate(le.classes_)},
    label2id={l:i for i,l in enumerate(le.classes_)},
    problem_type="single_label_classification",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=USE_FAST)

# =================== Dataset ===================
class DS(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None, tok=None, max_len=MAX_LEN):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, max_length=self.max_len, padding=False)
        if self.labels is not None:
            enc["labels"] = int(self.labels[i])
        return enc

# =================== Mean-Pool + Multi-Sample Dropout Head ===================
class DebertaMSD(nn.Module):
    def __init__(self, model_dir, config, n_drop=N_DROPOUT_PASSES, drop_p=HEAD_DROPOUT_P):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir, config=config, local_files_only=True)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(drop_p)
        self.n_drop = n_drop
        self.classifier = nn.Linear(hidden, config.num_labels)
    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.backbone(**batch)
        x = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        logits = 0
        for _ in range(self.n_drop):
            logits = logits + self.classifier(self.dropout(x))
        logits = logits / self.n_drop
        if labels is None:
            return type("obj", (), {"logits": logits})
        return type("obj", (), {"logits": logits, "labels": labels})

# =================== Class Weights (long tail) ===================
counts = np.bincount(y, minlength=num_labels).astype(np.float32)
inv = 1.0 / np.maximum(counts, 1.0)
weights = inv * (num_labels / inv.sum())
class_weights = torch.tensor(weights, dtype=torch.float)

# =================== Label-Smoothed, Class-Weighted Loss ===================
class LSWeightedTrainer(Trainer):
    def __init__(self, *a, label_smoothing=LABEL_SMOOTH, **kw):
        super().__init__(*a, **kw)
        self.label_smoothing = label_smoothing
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        fwd_inputs = {k: v for k,v in inputs.items() if k!="labels"}
        outputs = model(**fwd_inputs)
        logits = outputs.logits
        n = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits).fill_(self.label_smoothing / (n - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs = torch.log_softmax(logits, dim=-1)
        w = class_weights.to(logits.device).unsqueeze(0)
        loss = -(true_dist * log_probs * w).sum(dim=1).mean()
        return (loss, outputs) if return_outputs else loss

# =================== Balanced Sampler ===================
def make_balanced_sampler(y_idx):
    cls_counts = np.bincount(y_idx, minlength=num_labels).astype(np.float32)
    sw = 1.0 / np.maximum(cls_counts[y_idx], 1.0)
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

# =================== CV Split ===================
skf = StratifiedKFold(n_splits=TARGET_FOLDS, shuffle=True, random_state=SEED)
all_splits = list(skf.split(train, y))
if RUN_FOLDS == 1:
    splits_to_run = [all_splits[FOLD_INDEX_TO_RUN]]
else:
    splits_to_run = all_splits[:RUN_FOLDS]

# =================== Data Collator & Args ===================
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=VAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WD,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available(),
    bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    seed=SEED,
    report_to=[],
)

# =================== Train + Predict ===================
from copy import deepcopy

te_ds_full = DS(test[TEXT_COL].astype(str).tolist(), None, tokenizer, MAX_LEN)

oof_logits = np.zeros((len(train), num_labels), dtype=np.float32)
test_logits_accum = np.zeros((len(test), num_labels), dtype=np.float32)
val_scores = []

for fold_id, (tr_idx, va_idx) in enumerate(splits_to_run):
    print(f"\n===== FOLD {fold_id+1}/{len(splits_to_run)} =====")
    df_tr = train.iloc[tr_idx].reset_index(drop=True)
    df_va = train.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y[tr_idx], y[va_idx]

    tr_ds = DS(df_tr[TEXT_COL].astype(str).tolist(), y_tr, tokenizer, MAX_LEN)
    va_ds = DS(df_va[TEXT_COL].astype(str).tolist(), y_va, tokenizer, MAX_LEN)

    sampler = make_balanced_sampler(y_tr)

    model = DebertaMSD(MODEL_DIR, config)
    trainer = LSWeightedTrainer(
        model=model,
        args=deepcopy(args),
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # override dataloader to plug sampler
    trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
        tr_ds, batch_size=TRAIN_BS, sampler=sampler, collate_fn=data_collator
    )

    trainer.train()

    # VAL
    with torch.no_grad():
        pred_va = trainer.predict(va_ds)
    logits_va = pred_va.predictions if hasattr(pred_va, "predictions") else pred_va[0]
    probs_va  = torch.tensor(logits_va).softmax(dim=-1).cpu().numpy()
    oof_logits[va_idx] = logits_va

    pred_va_1 = probs_va.argmax(axis=1)
    acc = accuracy_score(y_va, pred_va_1)
    eps = 1e-15
    ll  = log_loss(y_va, np.clip(probs_va, eps, 1.0), labels=list(range(num_labels)))
    val_scores.append((acc, ll))
    print(f"Fold {fold_id} — ACC: {acc:.4f} | LOGLOSS: {ll:.4f}")

    # TEST with light TTA: (a) normal, (b) lowercase
    with torch.no_grad():
        out_a = trainer.predict(te_ds_full)
    logits_a = out_a.predictions if hasattr(out_a, "predictions") else out_a[0]

    # lowercased TTA
    te_lower = test[TEXT_COL].str.lower().tolist()
    te_ds_lower = DS(te_lower, None, tokenizer, MAX_LEN)
    with torch.no_grad():
        out_b = trainer.predict(te_ds_lower)
    logits_b = out_b.predictions if hasattr(out_b, "predictions") else out_b[0]

    test_logits_accum += (logits_a + logits_b) / 2.0

# =================== Report CV and Build Submission ===================
if len(val_scores) > 0:
    mean_acc = np.mean([a for a,_ in val_scores]); mean_ll = np.mean([b for _,b in val_scores])
    print(f"\nCV — ACC: {mean_acc:.4f} | LOGLOSS: {mean_ll:.4f}")

# Final test probs
probs_te = torch.tensor(test_logits_accum / len(splits_to_run)).softmax(dim=-1).cpu().numpy()

# =================== TOP-3 STRINGS ===================
def topk_strings_from_probs(probs_row, le, k=3):
    idx = np.argsort(-probs_row)[:k]
    labs = [le.inverse_transform([i])[0] for i in idx]
    out, seen = [], set()
    for L in labs:
        if L not in seen:
            out.append(L); seen.add(L)
    return " ".join(out[:k])

top3_strings = [topk_strings_from_probs(row, le, 3) for row in probs_te]

# =================== SAVE SUBMISSION ===================
id_like_cols = {"row_id","id","example_id"}
sample = pd.read_csv(SAMPLE_SUB)
pred_cols = [c for c in sample.columns if c not in id_like_cols]
pred_col = pred_cols[0] if pred_cols else sample.columns[1]
sub = sample.copy()
sub[pred_col] = top3_strings
sub.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv with shape {sub.shape} (prediction column: {pred_col!r})")

# =================== SAVE ARTIFACTS ===================
np.save("label_classes.npy", le.classes_)
np.save("oof_logits.npy", oof_logits)
np.save("test_logits.npy", test_logits_accum)
print("Saved: label_classes.npy, oof_logits.npy, test_logits.npy")

# quick distribution peek
top1_test = probs_te.argmax(1)
print("\nTest top-1 label distribution (first 20):")
print(pd.Series(le.inverse_transform(top1_test)).value_counts().head(20))
# ===============================================================================
