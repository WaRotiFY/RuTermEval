#!/usr/bin/env python3
import json
import os
import spacy
from collections import defaultdict
from statistics import mean

# ----------------- ПУТИ (отредактируйте при необходимости) -----------------
MODEL_DIR = "./ruterm_finetuned_spancat"   # Путь к директории со spacy-моделью
DATA_FILE = "cl-ruterm3-sample.json"       # Путь к JSON с разметкой
# Если ваша модель сохраняла spans под ключом, отличным от дефолтного,
# укажите его здесь (например "sc"). Иначе оставьте None и ключ будет определён автоматически.
SPANS_KEY = None  # или "sc"
SAMPLE_LIMIT = 10  # сколько примеров пробовать при автоопределении spans_key
# ---------------------------------------------------------------------------

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def detect_spans_key(nlp, sample_texts=None):
    if "spancat" in nlp.pipe_names:
        sp = nlp.get_pipe("spancat")
        cfg_key = getattr(sp, "cfg", {}).get("spans_key", None)
        if cfg_key:
            return cfg_key
    if not sample_texts:
        return None
    for txt in sample_texts:
        doc = nlp(txt)
        try:
            keys = list(doc.spans.keys())
            if keys:
                return keys[0]
        except Exception:
            pass
    return None

def compute_metrics(model_dir, data_file, spans_key_arg=None, sample_limit=10):
    if not os.path.exists(model_dir):
        raise SystemExit(f"Model path not found: {model_dir}")
    if not os.path.exists(data_file):
        raise SystemExit(f"Data file not found: {data_file}")

    nlp = spacy.load(model_dir)
    print("Loaded model from:", model_dir)
    print("Pipeline:", nlp.pipe_names)

    data = load_data(data_file)
    print("Loaded data:", len(data), "documents")

    spans_key = spans_key_arg or detect_spans_key(nlp, [d["text"] for d in data[:sample_limit]])
    if not spans_key:
        raise SystemExit("Не удалось определить spans_key для spancat. Укажите его в переменной SPANS_KEY.")
    print("Используем spans_key =", spans_key)

    total_TP = total_FP = total_FN = 0
    per_label = defaultdict(lambda: {"TP":0, "FP":0, "FN":0})
    per_label_support = defaultdict(int)  # количество золотых для каждой метки (support)
    per_doc_f1s = []  # список F1 для каждого документа (avg per docs)

    for idx, item in enumerate(data):
        text = item["text"]
        gold_ann = item.get("label", [])

        gold_set = set()
        for start, end, label in gold_ann:
            if start < 0 or end > len(text) or start >= end:
                continue
            gold_set.add((start, end, label))
            per_label_support[label] += 1

        doc = nlp(text)
        pred_spans = []
        try:
            pred_spans = doc.spans.get(spans_key, []) if hasattr(doc, "spans") else []
        except Exception:
            try:
                keys = list(doc.spans.keys())
                if spans_key in keys:
                    pred_spans = doc.spans.get(spans_key, [])
                else:
                    pred_spans = []
            except Exception:
                pred_spans = []

        pred_set = set()
        for s in pred_spans:
            try:
                start_char = s.start_char
                end_char = s.end_char
                label = s.label_
                pred_set.add((start_char, end_char, label))
            except Exception:
                try:
                    pred_set.add((s.start_char, s.end_char, getattr(s, "label_", None)))
                except Exception:
                    continue

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        total_TP += len(tp_set)
        total_FP += len(fp_set)
        total_FN += len(fn_set)

        for (s,e,l) in tp_set:
            per_label[l]["TP"] += 1
        for (s,e,l) in fp_set:
            per_label[l]["FP"] += 1
        for (s,e,l) in fn_set:
            per_label[l]["FN"] += 1

        # Пер-док метрика: strict exact-match per document (start,end,label)
        doc_TP = len(tp_set)
        doc_FP = len(fp_set)
        doc_FN = len(fn_set)
        doc_prec = doc_TP / (doc_TP + doc_FP) if (doc_TP + doc_FP) > 0 else 0.0
        doc_rec = doc_TP / (doc_TP + doc_FN) if (doc_TP + doc_FN) > 0 else 0.0
        doc_f1 = (2 * doc_prec * doc_rec / (doc_prec + doc_rec)) if (doc_prec + doc_rec) > 0 else 0.0
        per_doc_f1s.append(doc_f1)

    # Общие метрики (micro)
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1_micro = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-label metrics + Macro F1 (average of per-label F1)
    label_results = []
    for label, stats in sorted(per_label.items(), key=lambda x: x[0]):
        TP = stats["TP"]
        FP = stats["FP"]
        FN = stats["FN"]
        p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        label_results.append((label, TP, FP, FN, p, r, f))

    macro_f1 = mean([r[6] for r in label_results]) if label_results else 0.0

    # Class-weighted F1: weight per-label F1 by support (number of gold instances for that label)
    total_support = sum(per_label_support.values())
    if total_support > 0:
        weighted_f1 = sum((per_label_support[label] * (next((r[6] for r in label_results if r[0]==label), 0.0)))
                          for label in per_label_support) / total_support
    else:
        weighted_f1 = 0.0

    # Avg per-doc F1 (simple average across documents)
    avg_per_doc_f1 = mean(per_doc_f1s) if per_doc_f1s else 0.0

    # Вывод
    print("\n=== Общие метрики (micro) ===")
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall    (micro): {recall:.4f}")
    print(f"F1 (micro):        {f1_micro:.4f}")

    print("\n=== Macro / Weighted / Avg-per-doc ===")
    print(f"Macro-average F1 (avg per label): {macro_f1:.4f}")
    print(f"Class-weighted F1 (label F1 weighted by support): {weighted_f1:.4f}")
    print(f"Avg per-doc F1 (mean F1 across documents): {avg_per_doc_f1:.4f}")

    print("\n=== Метрики по меткам (label) ===")
    for label, TP, FP, FN, p, r, f in label_results:
        support = per_label_support.get(label, 0)
        print(f"{label:20s} support={support:4d}  TP={TP:4d} FP={FP:4d} FN={FN:4d}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    return {
        "total": {"TP": total_TP, "FP": total_FP, "FN": total_FN, "precision": precision, "recall": recall, "f1_micro": f1_micro},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_per_doc_f1": avg_per_doc_f1,
        "per_label": per_label
    }

if __name__ == "__main__":
    print(f"Model dir: {MODEL_DIR}")
    print(f"Data file: {DATA_FILE}")
    if SPANS_KEY:
        print(f"Forced spans_key: {SPANS_KEY}")
    else:
        print("Spans_key: auto-detect")

    compute_metrics(MODEL_DIR, DATA_FILE, spans_key_arg=SPANS_KEY, sample_limit=SAMPLE_LIMIT)