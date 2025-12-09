import json
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
from spacy.training import Example
import random
import os
import math
from collections import Counter, defaultdict

# ----------------- Настройки (при необходимости редактируйте) -----------------
DATA_URL = "https://raw.githubusercontent.com/mlsa-iai-msu-lab/cl-ruterm3/master/cl-ruterm3-sample.json"
DATA_FILE = "cl-ruterm3-sample.json"

OUT_MODEL_DIR = "./ruterm_finetuned_spancat"
SPANS_KEY = "sc"            # ключ в doc.spans, по умолчанию "sc" (если модель была сохранена с другим ключом, поменяйте)
N_ITER = 40                 # число эпох
BATCH_SIZE = 8
RANDOM_SEED = 42

# Условия апсемплинга редких классов: целевой support и максимальный множитель
UPSAMPLE_TARGET_SUPPORT = 200
UPSAMPLE_MAX_MULTIPLIER = 6

# ---------------------------------------------------------------------------

random.seed(RANDOM_SEED)


def load_data_from_github():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Данные из локального файла")
        return data
    try:
        import requests
        r = requests.get(DATA_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Данные загружены с GitHub и сохранены локально")
        return data
    except Exception as e:
        print("Ошибка загрузки данных:", e)
        return None


def convert_to_spacy_format(data_items, text_key="text", labels_key="label", spans_key=SPANS_KEY):
    nlp = spacy.blank("ru")
    doc_bin = DocBin()
    skipped_annotations = 0
    total_annotations = 0

    for idx, item in enumerate(data_items):
        text = item[text_key]
        annotations = item[labels_key]

        doc = nlp.make_doc(text)
        spans = []

        for ann in annotations:
            total_annotations += 1
            start, end, label = ann
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
            else:
                skipped_annotations += 1
                print(f"Документ {idx}: Не удалось создать span для '{text[start:end]}' ({start}:{end})")

        try:
            if spans:
                doc.spans[spans_key] = spans
            doc_bin.add(doc)
        except Exception as e:
            print(f"Ошибка при добавлении документа {idx}: {e}")

    print(f"\n=== Статистика конвертации ===")
    print(f"Всего аннотаций: {total_annotations}, пропущено при сохранении в DocBin: {skipped_annotations}")

    return doc_bin


def count_label_support(data):
    counter = Counter()
    for item in data:
        for s, e, label in item.get("label", []):
            counter[label] += 1
    return counter


def upsample_documents_by_label(train_data, per_label_support,
                                target_support=UPSAMPLE_TARGET_SUPPORT,
                                max_multiplier=UPSAMPLE_MAX_MULTIPLIER):
    """
    Апсемплинг документов, которые содержат редкие лейблы.
    Для документа вычисляем максимальный множитель по всем лейблам в документе:
      multiplier = min(max_multiplier, ceil(target_support / support[label]))
    """
    new_train = []
    for item in train_data:
        labels_in_doc = set(label for _, _, label in item.get("label", []))
        multiplier = 1
        for l in labels_in_doc:
            sup = per_label_support.get(l, 0)
            if sup > 0 and sup < target_support:
                mult = math.ceil(target_support / sup)
                mult = min(mult, max_multiplier)
                multiplier = max(multiplier, mult)
        for _ in range(multiplier):
            new_train.append(item)
    print(f"Upsampled train size: {len(new_train)} (original {len(train_data)})")
    return new_train


def build_examples(nlp, data, spans_key=SPANS_KEY):
    examples = []
    skipped = 0
    for item in data:
        text = item["text"]
        annotations = item.get("label", [])
        doc = nlp.make_doc(text)
        spans_list = []
        for start, end, label in annotations:
            # Example.from_dict expects tuples (start,end) or (start,end,label)
            spans_list.append((start, end, label))
        if not spans_list:
            skipped += 1
            continue
        gold = {"spans": {spans_key: spans_list}}
        try:
            ex = Example.from_dict(doc, gold)
            examples.append(ex)
        except Exception as e:
            skipped += 1
            print(f"Не удалось создать Example (len={len(text)}): {e}")
    print(f"Создано {len(examples)} Examples (пропущено {skipped})")
    return examples


def evaluate_spancat(nlp, data, spans_key=SPANS_KEY):
    """
    Строгая exact-match оценка (start,end,label).
    Возвращает dict с метриками micro precision/recall/f1 и TP/FP/FN.
    """
    total_TP = total_FP = total_FN = 0
    for item in data:
        text = item["text"]
        gold_ann = item.get("label", [])
        gold_set = set((s, e, l) for s, e, l in gold_ann if 0 <= s < e <= len(text))

        doc = nlp(text)
        pred_spans = []
        try:
            # doc.spans может быть not dict-like в некоторых ситуациях, поэтому безопасно
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
                pred_set.add((s.start_char, s.end_char, s.label_))
            except Exception:
                continue

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        total_TP += len(tp_set)
        total_FP += len(fp_set)
        total_FN += len(fn_set)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"TP": total_TP, "FP": total_FP, "FN": total_FN, "precision": precision, "recall": recall, "f1": f1}


def train_with_transfer_learning(train_data, dev_data, n_iter=N_ITER, spans_key=SPANS_KEY):
    # try to load pretrained model, fallback to blank
    try:
        nlp = spacy.load("ru_core_news_md")
        print("Загружена модель 'ru_core_news_md' для transfer learning.")
    except OSError:
        print("Модель 'ru_core_news_md' не найдена — используем пустую модель 'ru'.")
        nlp = spacy.blank("ru")

    # ensure tok2vec is present
    if "tok2vec" not in nlp.pipe_names and "transformer" not in nlp.pipe_names:
        nlp.add_pipe("tok2vec", first=True)

    # create spancat if not present (with spans_key only)
    if "spancat" not in nlp.pipe_names:
        nlp.add_pipe("spancat", last=True, config={"spans_key": spans_key})
        spancat = nlp.get_pipe("spancat")
        try:
            setattr(spancat, "allow_overlap", True)
        except Exception:
            pass
        print(f"Добавлен spancat с spans_key='{spans_key}'")
    else:
        spancat = nlp.get_pipe("spancat")
        cfg_key = getattr(spancat, "cfg", {}).get("spans_key", None)
        if cfg_key and cfg_key != spans_key:
            print(f"Внимание: существующий spancat использует spans_key='{cfg_key}', а ожидаем '{spans_key}'. Используем '{cfg_key}'.")
            spans_key = cfg_key
        try:
            setattr(spancat, "allow_overlap", True)
        except Exception:
            pass

    # add labels
    labels = set()
    for item in train_data:
        for s, e, l in item.get("label", []):
            labels.add(l)
    for label in sorted(labels):
        try:
            spancat.add_label(label)
        except Exception:
            pass

    # Upsample rare classes by duplicating documents
    per_label_support = count_label_support(train_data)
    print("Label support (train):", dict(per_label_support))
    up_train_data = upsample_documents_by_label(train_data, per_label_support)

    # Build examples
    train_examples = build_examples(nlp, up_train_data, spans_key=spans_key)
    dev_examples = build_examples(nlp, dev_data, spans_key=spans_key)

    if not train_examples:
        raise SystemExit("Нет обучающих examples после подготовки. Проверьте аннотации.")

    # Initialize pipeline (provide examples generator)
    nlp.initialize(lambda: (ex for ex in train_examples))
    optimizer = nlp.resume_training()

    best_f1 = -1.0
    best_epoch = -1

    print("Начинаем обучение...")
    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}
        batches = spacy.util.minibatch(train_examples, size=BATCH_SIZE)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        # evaluate on dev set (use raw texts to go through full pipeline)
        metrics = evaluate_spancat(nlp, dev_data, spans_key=spans_key)
        f1 = metrics["f1"]
        print(f"Эпоха {epoch+1}/{n_iter}: loss={losses.get('spancat', 0.0):.4f}  dev_f1={f1:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
        # save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            nlp.to_disk(OUT_MODEL_DIR)
            print(f"  -> Сохранена лучшая модель (epoch {best_epoch}) в {OUT_MODEL_DIR}")

    print(f"Обучение завершено. Лучший dev F1={best_f1:.4f} на эпохе {best_epoch}")
    return spancat, nlp


def pretty_predict_and_print(nlp, samples, spans_key=SPANS_KEY, max_show=3):
    print("\n=== Результаты инференса ===")
    for i, sample in enumerate(samples[:max_show]):
        print(f"\n--- Пример {i} ---")
        doc = nlp(sample["text"])
        try:
            keys = list(doc.spans.keys())
        except Exception:
            try:
                keys = [k for k in doc.spans]
            except Exception:
                keys = []
        print("doc.spans keys:", keys)
        if keys:
            for k in keys:
                spans = doc.spans.get(k, [])
                print(f"spans under key '{k}': {len(spans)}")
                for s in spans:
                    print(f"  - '{s.text}' ({s.label_}) [{s.start_char}:{s.end_char}] tokens={len(s)}")
        else:
            print("doc.ents:", [(ent.text, ent.label_) for ent in doc.ents])
            print("Нет doc.spans ключей — возможно spancat не добавлял спаны или spans_key другой.")
        print("Ожидаемые термины (разметка):")
        for start, end, label in sample["label"][:20]:
            print(f"  - '{sample['text'][start:end]}' ({label}) [{start}:{end}]")


if __name__ == "__main__":
    data = load_data_from_github()
    if data is None:
        raise SystemExit("Данные не загружены. Останов.")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Тренировочные данные: {len(train_data)} документов")
    print(f"Тестовые данные: {len(test_data)} документов")

    # Сохраним DocBin (необязательно, но удобно)
    train_docbin = convert_to_spacy_format(train_data, spans_key=SPANS_KEY)
    test_docbin = convert_to_spacy_format(test_data, spans_key=SPANS_KEY)
    train_docbin.to_disk("ruterm_train.spacy")
    test_docbin.to_disk("ruterm_test.spacy")

    # Дообучаем spancat
    spancat, nlp = train_with_transfer_learning(train_data, test_data, n_iter=N_ITER, spans_key=SPANS_KEY)

    # Загружаем и показываем примеры предсказаний
    nlp = spacy.load(OUT_MODEL_DIR)
    pretty_predict_and_print(nlp, test_data[:3], spans_key=SPANS_KEY)
