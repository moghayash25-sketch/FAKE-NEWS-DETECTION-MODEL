!pip install -q transformers
import os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
# ══════════════════════════════════════════════════════════════════
# CONFIG — edit before running
# ══════════════════════════════════════════════════════════════════
MODE = 'batch'   # 'single' | 'batch'

# ── Mode A ────────────────────────────────────────────────────────
SAMPLE_TEXT = "NASA confirms water found on the moon's sunlit surface"

# ── Mode B ────────────────────────────────────────────────────────
BATCH_TSV  = '/kaggle/input/datasets/yashmogha/dataset/multimodal_test_public.tsv'
BATCH_SIZE = 128     # larger than training is fine — inference only
SAVE_CSV   = '/kaggle/working/text_predictions.csv'

# ── Checkpoint ────────────────────────────────────────────────────
TEXT_CKPT = '/kaggle/input/datasets/yashmogha/dataset/text_best.pt'

# ── Must match training exactly ───────────────────────────────────
CFG = {
    'model_name':    'bert-base-uncased',
    'num_labels':    6,
    'freeze_layers': 8,
    'dropout':       0.3,
    'max_text_len':  128,
    'label_col':     '6_way_label',
    'out_dir':       '/kaggle/working',

    'class_names': [
        'true',
        'satire/parody',
        'misleading content',
        'imposter content',
        'false connection',
        'manipulated content',
    ],
    'class_colors': [
        '#2ecc71', '#f39c12', '#e74c3c',
        '#8e44ad', '#2980b9', '#c0392b',
    ],
}

os.makedirs(CFG['out_dir'], exist_ok=True)
print('Config loaded.')
class TextModel(nn.Module):
    """
    Exact replica of the training architecture.
    BERT → [CLS] → Dropout → Linear → num_labels
    """
    def __init__(self, num_labels=6, model_name='bert-base-uncased',
                 dropout=0.3, freeze_layers=8):
        super().__init__()
        self.bert      = BertModel.from_pretrained(model_name)
        self.embed_dim = self.bert.config.hidden_size   # 768
        self.drop      = nn.Dropout(dropout)
        self.head      = nn.Linear(self.embed_dim, num_labels)
        # freeze same layers as during training so weights load cleanly
        if freeze_layers > 0:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head(self.drop(cls))

    model = TextModel(
        num_labels=CFG['num_labels'],
        model_name=CFG['model_name'],
        dropout=CFG['dropout'],
        freeze_layers=CFG['freeze_layers'],
    ).to(DEVICE)

    model.load_state_dict(torch.load(TEXT_CKPT, map_location=DEVICE))
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f'✓ Model loaded from {TEXT_CKPT}')
    print(f'  Params: {total / 1e6:.1f}M')
# ── Tokenizer ─────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(CFG['model_name'])

def encode(texts):
    enc = tokenizer(
        texts,
        max_length     = CFG['max_text_len'],
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt',
    )
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)

print('Tokenizer ready.')
@torch.no_grad()
def predict(texts):
    """
    texts : list[str]

    Returns:
        preds      : list[str]        — class name per sample
        probs      : np.ndarray (N,6) — softmax probabilities
        confidence : list[float]      — max prob per sample
    """
    ids, mask = encode(texts)
    logits    = model(ids, mask)                          # (N, 6)
    probs     = F.softmax(logits, dim=1).cpu().numpy()
    preds     = [CFG['class_names'][p] for p in probs.argmax(axis=1)]
    conf      = probs.max(axis=1).tolist()
    return preds, probs, conf
if MODE == 'single':
    preds, probs, conf = predict([SAMPLE_TEXT])

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(CFG['class_names'], probs[0], color=CFG['class_colors'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(f'"{SAMPLE_TEXT[:80]}"', fontsize=9)
    for bar, v in zip(bars, probs[0]):
        ax.text(min(v + 0.01, 0.93), bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=9)
    pred_color = CFG['class_colors'][CFG['class_names'].index(preds[0])]
    plt.suptitle(f'Prediction: {preds[0].upper()}  ({conf[0]:.1%} confidence)',
                 color=pred_color, fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['out_dir'], 'single_prediction.png'), dpi=150)
    plt.show()

    print(f'\nPrediction : {preds[0]}')
    print(f'Confidence : {conf[0]:.1%}')
    print('\nAll probabilities:')
    for name, p in zip(CFG['class_names'], probs[0]):
        print(f'  {name:<25} {"█" * int(p * 30):<30} {p:.4f}')
if MODE == 'batch':
    df       = pd.read_csv(BATCH_TSV, sep='\t', dtype=str)
    text_col = 'clean_title' if 'clean_title' in df.columns else 'title'
    df[text_col] = df[text_col].fillna('')
    print(f'Loaded {len(df):,} rows  |  text col: "{text_col}"')

    all_preds, all_conf, all_probs = [], [], []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc='Predicting'):
        texts = df[text_col].iloc[start : start + BATCH_SIZE].tolist()
        p, prb, c = predict(texts)
        all_preds.extend(p)
        all_conf.extend(c)
        all_probs.extend(prb.tolist())

    # Build results dataframe
    results = df[['id', text_col]].copy()
    if CFG['label_col'] in df.columns:
        results['true_label'] = df[CFG['label_col']].apply(
            lambda x: CFG['class_names'][int(x)] if pd.notna(x) else 'unknown'
        )
    results['predicted']  = all_preds
    results['confidence'] = [f'{c:.4f}' for c in all_conf]
    for i, name in enumerate(CFG['class_names']):
        col = f'prob_{name.replace("/","_").replace(" ","_")}'
        results[col] = [p[i] for p in all_probs]

    results.to_csv(SAVE_CSV, index=False)
    print(f'\n✓ Saved {len(results):,} predictions → {SAVE_CSV}')
    display(results.head(10))
if MODE == 'batch':
    # ── Accuracy & classification report (if ground truth available) ─
    if 'true_label' in results.columns:
        known    = results[results['true_label'] != 'unknown']
        correct  = (known['true_label'] == known['predicted']).sum()
        true_idx = [CFG['class_names'].index(l) for l in known['true_label']]
        pred_idx = [CFG['class_names'].index(l) for l in known['predicted']]

        print(f'Accuracy : {correct}/{len(known)} = {correct/len(known):.4f}')
        print(f'Macro-F1 : {f1_score(true_idx, pred_idx, average="macro"):.4f}')
        print()
        print(classification_report(true_idx, pred_idx,
                                     target_names=CFG['class_names'], digits=4))
if MODE == 'batch':
    # ── Prediction distribution + confidence histogram ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pred_counts = pd.Series(all_preds).value_counts().reindex(
        CFG['class_names'], fill_value=0
    )
    axes[0].bar(pred_counts.index, pred_counts.values, color=CFG['class_colors'])
    axes[0].set_title('Prediction Distribution')
    axes[0].set_ylabel('Count')
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right')
    for i, v in enumerate(pred_counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontsize=8)

    axes[1].hist(all_conf, bins=25, color='steelblue', edgecolor='white')
    axes[1].axvline(np.mean(all_conf), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_conf):.2f}')
    axes[1].set_title('Confidence Distribution')
    axes[1].set_xlabel('Confidence'); axes[1].legend()

    plt.suptitle(f'Text Model Predictions — {len(df):,} samples')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['out_dir'], 'text_prediction_summary.png'), dpi=150)
    plt.show()

    # ── Low-confidence samples ─────────────────────────────────────
    results['_conf_float'] = all_conf
    uncertain = results[results['_conf_float'] < 0.5].sort_values('_conf_float')
    print(f'Low-confidence predictions (< 50%): {len(uncertain):,}')
    display(uncertain[['id', text_col, 'predicted', 'confidence']].head(10))
shutil.make_archive('/kaggle/working/text_inference_outputs', 'zip', '/kaggle/working')
print('✓ text_inference_outputs.zip ready — download from Output tab')