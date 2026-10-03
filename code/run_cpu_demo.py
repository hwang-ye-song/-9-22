# -*- coding: utf-8 -*-
"""00·01·02 실습 핵심 단계를 CPU로 재현하고 그림을 저장한다 (원본: yblee110/vision-ai-notebook)."""
import hashlib, io, json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

ROOT = Path(__file__).parent  # models/, cache/, out/ 이 이 폴더 아래에 생깁니다
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "models/deit-tiny"
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def section(t): log(""); log(f"===== {t} =====")

torch.manual_seed(42)
torch.set_num_threads(max(1, torch.get_num_threads()))
device = torch.device("cpu")
t0 = time.perf_counter()

# ---------------- 00: hf download ----------------
section("00 · hf download")
MODEL_ID = "facebook/deit-tiny-patch16-224"
MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
MODEL_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
EXPECTED = {'config.json': 'e8786a68c993c04b8721c4d379e4f9de6a9564fba5987704dbf8696d53b7972d',
            'preprocessor_config.json': '90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2',
            'pytorch_model.bin': 'e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c'}
HF = next((c for c in (Path(sys.executable).parent / "hf.exe", Path(sys.executable).parent / "hf") if c.is_file()), Path(shutil.which("hf") or "hf"))
cmd = [str(HF), "download", MODEL_ID, *MODEL_FILES, "--revision", MODEL_REVISION, "--local-dir", str(MODEL_DIR)]
log("$ hf download", MODEL_ID, *MODEL_FILES, "--revision", MODEL_REVISION, "--local-dir models/deit-tiny")
subprocess.run(cmd, check=True, timeout=600, capture_output=True)
for f in MODEL_FILES:
    p = MODEL_DIR / f
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == EXPECTED[f], f
    log(f"{f:<26} {p.stat().st_size:>10,} bytes · SHA256 확인")

# ---------------- 00: 데이터 준비 ----------------
section("00 · CIFAR-100 → 5개 클래스 분할")
CLASSES = ["bottle", "bowl", "can", "cup", "plate"]
FINE = [9, 10, 16, 28, 61]
DATASET_ID = "uoft-cs/cifar100"; DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
SRC = {"train": "694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23",
       "test": "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"}
def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
paths = {}
for split in ("train", "test"):
    fn = f"{split}-00000-of-00001.parquet"; target = CACHE / fn
    if not target.is_file():
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/cifar100/{fn}"
        with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as w: shutil.copyfileobj(r, w)
    assert digest(target) == SRC[split]; paths[split] = target
    log("Verified:", fn)
tr = pq.read_table(paths["train"], columns=["img", "fine_label"]); te = pq.read_table(paths["test"], columns=["img", "fine_label"])
trl = tr["fine_label"].to_numpy(); tel = te["fine_label"].to_numpy()
rng = np.random.default_rng(42); tri, vai = [], []
for lab in FINE:
    sh = rng.permutation(np.flatnonzero(trl == lab)); tri.extend(sh[:100]); vai.extend(sh[100:120])
rng = np.random.default_rng(43)
tei = np.concatenate([rng.permutation(np.flatnonzero(tel == lab))[:40] for lab in FINE])
lm = {o: n for n, o in enumerate(FINE)}
splits = {}
for name, table, labels, idx in [("train", tr, trl, tri), ("validation", tr, trl, vai), ("test", te, tel, tei)]:
    px = [np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in table.take([int(i) for i in idx])["img"].to_pylist()]
    splits[name] = {"images": np.stack(px), "labels": np.asarray([lm[int(labels[i])] for i in idx], dtype=np.int64)}
for name, s in splits.items():
    log(f"{name:<10}", dict(zip(CLASSES, np.bincount(s["labels"], minlength=5).tolist())), s["images"].shape)

fig, axes = plt.subplots(3, 5, figsize=(10, 6.4))
for c, name in enumerate(CLASSES):
    idx = np.flatnonzero(splits["train"]["labels"] == c)[:3]
    for r, i in enumerate(idx):
        axes[r, c].imshow(splits["train"]["images"][i], interpolation="nearest"); axes[r, c].axis("off")
    axes[0, c].set_title(name, fontsize=13)
fig.suptitle("Training samples · CIFAR-100 5 classes · 32×32 pixels", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "01_data_samples.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------- 01: 추론 ----------------
section("01 · 원본 모델 추론 (ImageNet 1,000 라벨)")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True, use_safetensors=False,
                                                        weights_only=True, attn_implementation="eager").to(device)
log(type(model).__name__, "· 원래 분류 수:", model.config.num_labels)
log("전체 파라미터:", f"{sum(p.numel() for p in model.parameters()):,}")

cup = CLASSES.index("cup")
cup_idx = int(np.flatnonzero(splits["test"]["labels"] == cup)[0])
image = Image.fromarray(splits["test"]["images"][cup_idx]).convert("RGB")
model.eval()
clf = pipeline("image-classification", model=model, image_processor=processor, framework="pt", device=device)
log("[pipeline] 실제 정답: cup")
for item in clf(image, top_k=5):
    log(f"  {item['score']:6.2%}  {item['label']}")

def infer_probabilities(model, processor, image, device):
    inputs = {n: t.to(device) for n, t in processor(images=image, return_tensors="pt").items()}
    model.eval()
    with torch.inference_mode():
        return model(**inputs).logits.softmax(dim=-1)[0]
def topk_predictions(probabilities, id2label, k):
    values, indices = probabilities.topk(k)
    return [{"label": id2label[int(i)], "probability": float(v)} for v, i in zip(values, indices)]
probs = infer_probabilities(model, processor, image, device)
log("[infer_probabilities] 개수:", probs.numel(), "· 합:", round(float(probs.sum()), 4))
preds = topk_predictions(probs, model.config.id2label, 5)
for k in (1, 3):
    log(f"[topk k={k}]", ", ".join(f"{p['label'].split(',')[0]} {p['probability']:.1%}" for p in topk_predictions(probs, model.config.id2label, k)))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].imshow(image); axes[0].set_title("Input: cup (32×32)"); axes[0].axis("off")
labels = [p["label"].split(",")[0] for p in preds]
axes[1].barh(labels[::-1], [p["probability"] for p in preds][::-1], color="#1a73e8")
axes[1].set_xlim(0, 1); axes[1].set_xlabel("Predicted probability"); axes[1].set_title("Pretrained ImageNet model: top 5")
fig.savefig(OUT / "02_pretrained_top5.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------- 02: 파인튜닝 ----------------
section("02 · 파인튜닝 (CPU)")
BATCH, HEAD_EPOCHS, FT_EPOCHS, HEAD_LR, FT_LR = 16, 3, 3, 1e-3, 1e-4
loaders = {}
for name, s in splits.items():
    pix = processor(images=[Image.fromarray(x).convert("RGB") for x in s["images"]], return_tensors="pt")["pixel_values"]
    loaders[name] = DataLoader(TensorDataset(pix, torch.tensor(s["labels"])), batch_size=BATCH,
                               shuffle=(name == "train"), generator=torch.Generator().manual_seed(42), num_workers=0)
log("입력 크기:", tuple(loaders["train"].dataset.tensors[0].shape))
for p in model.parameters(): p.requires_grad = False
model.classifier = nn.Linear(model.config.hidden_size, 5)
nn.init.normal_(model.classifier.weight, std=model.config.initializer_range); nn.init.zeros_(model.classifier.bias)
model.num_labels = 5; model.config.num_labels = 5
model.config.id2label = dict(enumerate(CLASSES)); model.config.label2id = {c: i for i, c in enumerate(CLASSES)}
criterion = nn.CrossEntropyLoss()

def train_one_batch(model, optimizer, criterion, pixels, labels):
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(pixel_values=pixels).logits; loss = criterion(logits, labels)
    correct = int((logits.argmax(dim=-1) == labels).sum()); count = int(len(labels))
    loss.backward(); optimizer.step()
    return float(loss.item()), correct, count
def select_finetune_parameters(model, scope):
    for p in model.parameters(): p.requires_grad_(False)
    mods = [model.classifier]
    if scope == "last_block": mods.extend([model.vit.encoder.layer[-1], model.vit.layernorm])
    for m in mods:
        for p in m.parameters(): p.requires_grad_(True)
    return {n: p for n, p in model.named_parameters() if p.requires_grad}
def evaluate(loader):
    model.eval(); ls, L, P = 0.0, [], []
    with torch.inference_mode():
        for x, y in loader:
            lg = model(pixel_values=x).logits; ls += criterion(lg, y).item() * len(y)
            L += y.tolist(); P += lg.argmax(-1).tolist()
    L, P = np.asarray(L), np.asarray(P); m = np.zeros((5, 5), int); np.add.at(m, (L, P), 1)
    d = m.sum(0) + m.sum(1); f1 = np.divide(2 * m.diagonal(), d, out=np.zeros(5), where=d > 0)
    return {"loss": ls / len(L), "accuracy": float((L == P).mean()), "macro_f1": float(f1.mean()), "cm": m}

for scope in ("head", "last_block"):
    log(f"select_finetune_parameters('{scope}') →", f"{sum(p.numel() for p in select_finetune_parameters(model, scope).values()):,}개")

history = []
def run_stage(stage, params, lr, epochs):
    opt = torch.optim.Adam(params, lr=lr); best = ((-1.0, float("-inf")), None, None)
    tracked = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for ep in range(1, epochs + 1):
        ts = time.perf_counter(); tl, tc, tn = 0.0, 0, 0
        for x, y in loaders["train"]:
            l, c, n = train_one_batch(model, opt, criterion, x, y); tl += l * n; tc += c; tn += n
        v = evaluate(loaders["validation"])
        history.append({"stage": stage, "epoch": ep, "train_loss": tl / tn, "train_accuracy": tc / tn,
                        "validation_loss": v["loss"], "validation_accuracy": v["accuracy"]})
        log(f"{stage} {ep}/{epochs} · train loss {tl/tn:.3f} · train acc {tc/tn:.1%} · validation acc {v['accuracy']:.1%} · {time.perf_counter()-ts:.0f}s")
        key = (v["accuracy"], -v["loss"])
        if key > best[0]: best = (key, ep, {n: p.detach().clone() for n, p in tracked})
    with torch.no_grad():
        for n, p in tracked: p.copy_(best[2][n])
    return best[1]

select_finetune_parameters(model, "head")
head_ep = run_stage("head", model.classifier.parameters(), HEAD_LR, HEAD_EPOCHS)
head_test = evaluate(loaders["test"]); log(f"선택 epoch {head_ep} · head test {head_test['accuracy']:.1%} · macro F1 {head_test['macro_f1']:.3f}")
trainable = select_finetune_parameters(model, "last_block")
ft_ep = run_stage("finetune", list(trainable.values()), FT_LR, FT_EPOCHS)
ft_test = evaluate(loaders["test"]); log(f"선택 epoch {ft_ep} · fine-tune test {ft_test['accuracy']:.1%} · macro F1 {ft_test['macro_f1']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for stage, color in (("head", "#1a73e8"), ("finetune", "#188038")):
    rows = [r for r in history if r["stage"] == stage]; e = [r["epoch"] for r in rows]
    axes[0].plot(e, [r["train_loss"] for r in rows], "o-", color=color, label=f"{stage}: train")
    axes[0].plot(e, [r["validation_loss"] for r in rows], "s--", color=color, label=f"{stage}: validation")
    axes[1].plot(e, [r["validation_accuracy"] for r in rows], "o-", color=color, label=stage)
axes[0].set(title="Loss by training stage (CPU run)", xlabel="Epoch", ylabel="Cross-entropy loss")
axes[1].set(title="Validation accuracy (100 images)", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
for a in axes: a.legend(fontsize=9); a.grid(alpha=.2); a.set_xticks([1, 2, 3])
fig.savefig(OUT / "03_learning_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
mx = max(head_test["cm"].max(), ft_test["cm"].max())
for a, title, m in zip(axes, ("Head only", "Last block fine-tune"), (head_test, ft_test)):
    im = a.imshow(m["cm"], vmin=0, vmax=mx, cmap="Blues")
    a.set_xticks(range(5), CLASSES, rotation=35, ha="right"); a.set_yticks(range(5), CLASSES)
    a.set(title=f"{title}: {m['accuracy']:.1%} (200 test images)", xlabel="Predicted", ylabel="True")
    for r, c in np.ndindex(m["cm"].shape):
        a.text(c, r, str(m["cm"][r, c]), ha="center", va="center", color="white" if m["cm"][r, c] > mx / 2 else "black")
fig.colorbar(im, ax=axes, label="Image count", shrink=.85)
fig.savefig(OUT / "04_confusion_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

model.eval()
with torch.inference_mode():
    pr = model(**processor(images=image, return_tensors="pt")).logits.softmax(-1)[0]
log("[파인튜닝 후 같은 컵 이미지]", ", ".join(f"{CLASSES[i]} {float(pr[i]):.1%}" for i in pr.argsort(descending=True)))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].imshow(image); axes[0].set_title("Same input: cup (32×32)"); axes[0].axis("off")
order = pr.argsort(descending=True).tolist()
axes[1].barh([CLASSES[i] for i in order][::-1], [float(pr[i]) for i in order][::-1], color="#188038")
axes[1].set_xlim(0, 1); axes[1].set_xlabel("Predicted probability"); axes[1].set_title("After fine-tuning: 5 classes")
fig.savefig(OUT / "05_finetuned_cup.png", dpi=150, bbox_inches="tight"); plt.close(fig)

summary = {"device": "cpu", "torch": torch.__version__, "elapsed_seconds": round(time.perf_counter() - t0),
           "head": {"epoch": head_ep, "test_accuracy": head_test["accuracy"], "macro_f1": head_test["macro_f1"]},
           "finetune": {"epoch": ft_ep, "test_accuracy": ft_test["accuracy"], "macro_f1": ft_test["macro_f1"]},
           "history": history}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
log(""); log(f"총 실행 시간: {summary['elapsed_seconds']}초 (CPU)")
(OUT / "log.txt").write_text("\n".join(LOG), encoding="utf-8")
