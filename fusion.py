import os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
CFG = {

    'text_embeds_dir':  '/kaggle/input/datasets/yashmogha/dataset/textembed',
    'image_embeds_dir': '/kaggle/input/datasets/yashmogha/dataset/Archive 2',
    'out_dir':          '/kaggle/working',


    'text_dim':   768,
    'image_dim':  2048,


    'num_labels':  6,
    'hidden_dims': [512, 128],
    'dropout':     0.5,

    'epochs':          20,
    'patience':        3,
    'batch_size':      256,
    'lr':              3e-4,
    'weight_decay':    1e-2,
    'label_smoothing': 0.1,

    'class_names': [
        'true', 'satire/parody', 'misleading content',
        'imposter content', 'false connection', 'manipulated content',
    ],
}

TEXT_EMBED_FILES = {
    'train': '01_text_train_embeds.pt',
    'val':   '02_text_val_embeds.pt',
    'test':  '03_text_test_embeds.pt',
}
IMAGE_EMBED_FILES = {
    'train': '01_image_train_embeds.pt',
    'val':   '02_image_val_embeds.pt',
    'test':  '03_image_test_embeds.pt',
}

CKPT = os.path.join(CFG['out_dir'], 'fusion_best.pt')
print('Config loaded.')
print(f'Fusion input dim: {CFG["text_dim"]} + {CFG["image_dim"]} = {CFG["text_dim"] + CFG["image_dim"]}')
def load_and_align(text_embeds_dir, image_embeds_dir, split):

    text_path  = os.path.join(text_embeds_dir,  TEXT_EMBED_FILES[split])
    image_path = os.path.join(image_embeds_dir, IMAGE_EMBED_FILES[split])

    text_data  = torch.load(text_path,  map_location='cpu')
    image_data = torch.load(image_path, map_location='cpu')

    text_emb  = text_data['embeds']
    image_emb = image_data['embeds']
    n_text    = text_emb.shape[0]
    n_image   = image_emb.shape[0]

    print(f'  [{split}] text: {n_text:,}  image: {n_image:,}', end='  ')

    if n_text == n_image:
        print('→ sizes match, no alignment needed')
        return text_emb, image_emb, text_data['labels']

    assert 'row_indices' in image_data, (
        f"'row_indices' not found in {image_path}.\n"
        f'Re-run Section 5 of the image model notebook to regenerate embeddings.'
    )

    row_indices      = image_data['row_indices']
    text_emb_aligned = text_emb[row_indices]
    labels_aligned   = text_data['labels'][row_indices]

    if 'labels' in image_data:
        mismatches = (labels_aligned != image_data['labels']).sum().item()
        status = f'✓ labels match' if mismatches == 0 else f'⚠ {mismatches} mismatches!'
    else:
        status = ''

    print(f'→ aligned: {n_image:,} samples  '
          f'(dropped {n_text - n_image:,} text-only rows)  {status}')
    return text_emb_aligned, image_emb, labels_aligned


print('Loading embeddings...')
train_text, train_img, train_y = load_and_align(CFG['text_embeds_dir'], CFG['image_embeds_dir'], 'train')
val_text,   val_img,   val_y   = load_and_align(CFG['text_embeds_dir'], CFG['image_embeds_dir'], 'val')
test_text,  test_img,  test_y  = load_and_align(CFG['text_embeds_dir'], CFG['image_embeds_dir'], 'test')
print('Done.')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, labels, title in zip(axes, [train_y, val_y, test_y], ['Train', 'Val', 'Test']):
    counts = torch.bincount(labels, minlength=CFG['num_labels']).numpy()
    ax.bar(CFG['class_names'], counts, color='mediumpurple')
    ax.set_title(f'{title} ({labels.shape[0]:,} samples)')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    for i, v in enumerate(counts):
        ax.text(i, v + 20, str(v), ha='center', fontsize=7)
plt.suptitle('Class distribution after text–image alignment')
plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png', dpi=150)
plt.show()


y_train_np = train_y.numpy()

class_weights_np = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.arange(CFG['num_labels']),
    y            = y_train_np,
)

class_weights = torch.tensor(class_weights_np, dtype=torch.float)

print('Balanced class weights:')
for name, w, count in zip(CFG['class_names'], class_weights_np,
                           np.bincount(y_train_np, minlength=CFG['num_labels'])):
    print(f'  {name:<25} count: {count:>7,}   weight: {w:.4f}')
def build_loaders(train_text, train_img, train_y,
                  val_text,   val_img,   val_y,
                  test_text,  test_img,  test_y, cfg):
    kw = dict(batch_size=cfg['batch_size'], pin_memory=True)

    train_dl = DataLoader(TensorDataset(train_text, train_img, train_y), shuffle=True,  **kw)
    val_dl   = DataLoader(TensorDataset(val_text,   val_img,   val_y),   shuffle=False, **kw)
    test_dl  = DataLoader(TensorDataset(test_text,  test_img,  test_y),  shuffle=False, **kw)

    print(f'Batches → train: {len(train_dl)}  val: {len(val_dl)}  test: {len(test_dl)}')
    return train_dl, val_dl, test_dl


train_dl, val_dl, test_dl = build_loaders(
    train_text, train_img, train_y,
    val_text,   val_img,   val_y,
    test_text,  test_img,  test_y,
    CFG
)
class FeatureGatedFusion(nn.Module):

    def __init__(self, text_dim, image_dim, hidden_dims, num_labels, dropout):
        super().__init__()
        fused_dim = text_dim + image_dim

        self.feature_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 4),
            nn.GELU(),
            nn.Linear(fused_dim // 4, fused_dim),
            nn.Sigmoid()
        )
        layers = [nn.LayerNorm(fused_dim)]
        in_dim = fused_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim  = h
        layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, text_x, img_x):
        x            = torch.cat([text_x, img_x], dim=1)   # (B, 2816)
        gate_weights = self.feature_gate(x)                 # (B, 2816)  0–1 mask
        return self.classifier(x * gate_weights)            # (B, 6)


model = FeatureGatedFusion(
    text_dim    = CFG['text_dim'],
    image_dim   = CFG['image_dim'],
    hidden_dims = CFG['hidden_dims'],
    num_labels  = CFG['num_labels'],
    dropout     = CFG['dropout'],
).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
print(model)
print(f'\nParams: {total/1e6:.2f}M  (MLP only — BERT+ResNet frozen, not trained here)')
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for text_x, img_x, labels in tqdm(loader, desc='train', leave=False):
        text_x, img_x, labels = text_x.to(DEVICE), img_x.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=DEVICE.type == 'cuda'):
            logits = model(text_x, img_x)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_sum += loss.item() * labels.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += labels.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, desc='val'):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for text_x, img_x, labels in tqdm(loader, desc=desc, leave=False):
        text_x, img_x, labels = text_x.to(DEVICE), img_x.to(DEVICE), labels.to(DEVICE)
        logits = model(text_x, img_x)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        loss_sum += loss.item() * labels.size(0)
        correct  += (preds == labels).sum().item(); n += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return loss_sum / n, correct / n, all_preds, all_labels
optimizer    = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
total_steps  = CFG['epochs'] * len(train_dl)
warmup_steps = total_steps // 10
scheduler    = get_cosine_schedule_with_warmup(optimizer,
                   num_warmup_steps=warmup_steps, num_training_steps=total_steps)


criterion = nn.CrossEntropyLoss(
    weight          = class_weights.to(DEVICE),
    label_smoothing = CFG['label_smoothing'],
)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

print(f'Total steps: {total_steps} | Warmup: {warmup_steps}')
print(f'Loss: CrossEntropyLoss(balanced weights, label_smoothing={CFG["label_smoothing"]})')
print(f'Early stopping patience: {CFG["patience"]} epochs')

best_macro_f1    = 0.0
best_val_acc     = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

for epoch in range(1, CFG['epochs'] + 1):
    tl, ta               = train_epoch(model, train_dl, optimizer, criterion, scaler, scheduler)
    vl, va, vp, vl_true  = eval_epoch(model, val_dl, criterion)
    macro_f1 = f1_score(vl_true, vp, average='macro')

    history['train_loss'].append(tl); history['train_acc'].append(ta)
    history['val_loss'].append(vl);   history['val_acc'].append(va)
    history['val_f1'].append(macro_f1)

    flag = ''
    if macro_f1 > best_macro_f1:
        best_macro_f1    = macro_f1
        best_val_acc     = va
        patience_counter = 0
        torch.save(model.state_dict(), CKPT)
        flag = '  ✓ saved'
    else:
        patience_counter += 1
        flag = f'  (patience {patience_counter}/{CFG["patience"]})'

    print(f'Epoch {epoch:02d}/{CFG["epochs"]}  '
          f'train {tl:.4f}/{ta:.4f}  '
          f'val {vl:.4f}/{va:.4f}  '
          f'macro_f1 {macro_f1:.4f}  '
          f'lr {scheduler.get_last_lr()[0]:.2e}{flag}')

    if patience_counter >= CFG['patience']:
        print(f'\nEarly stopping triggered at epoch {epoch}.')
        break

print(f'\nBest val macro-F1: {best_macro_f1:.4f}  |  Best val acc: {best_val_acc:.4f}')
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
eps = range(1, len(history['train_loss']) + 1)
axes[0].plot(eps, history['train_loss'], label='train')
axes[0].plot(eps, history['val_loss'],   label='val')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(eps, history['train_acc'], label='train')
axes[1].plot(eps, history['val_acc'],   label='val')
axes[1].set_title('Accuracy'); axes[1].legend()
axes[2].plot(eps, history['val_f1'], color='darkorange')
axes[2].set_title('Val Macro-F1')
plt.suptitle('Fusion Model — Training History')
plt.tight_layout()
plt.savefig('/kaggle/working/fusion_history.png', dpi=150)
plt.show()
def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm      = confusion_matrix(labels, preds).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, t in zip(axes, [cm.astype(int), cm_norm],
                                ['d', '.2f'], ['Raw Counts', 'Normalised']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Purples',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{title} — {t}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.show()


def full_eval(model, loader, criterion, split_name, save_prefix):
    loss, acc, preds, labels = eval_epoch(model, loader, criterion, desc=split_name)
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f'\n{"="*60}')
    print(f'{split_name.upper()} SET  —  loss: {loss:.4f}  acc: {acc:.4f}  macro-F1: {macro_f1:.4f}')
    print(f'{"="*60}\n')
    print(classification_report(labels, preds, target_names=CFG['class_names'], digits=4))
    plot_confusion_matrix(labels, preds, CFG['class_names'],
                          f'Fusion ({split_name})',
                          f'/kaggle/working/{save_prefix}_cm.png')
    return loss, acc, macro_f1, preds, labels
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
criterion_eval = nn.CrossEntropyLoss(
    weight          = class_weights.to(DEVICE),
    label_smoothing = CFG['label_smoothing'],
)
print(f'Loaded best checkpoint  (val macro-F1: {best_macro_f1:.4f})')
val_loss, val_acc, val_f1, _, _ = full_eval(
    model, val_dl, criterion_eval, 'Validation', 'fusion_val'
)
test_loss, test_acc, test_f1, test_preds, test_labels = full_eval(
    model, test_dl, criterion_eval, 'Test', 'fusion_test'
)
# ── Per-class F1 bar chart ─────────────────────────────────────────
per_class_f1 = f1_score(test_labels, test_preds, average=None)

fig, ax = plt.subplots(figsize=(11, 5))
colors = ['#2ecc71','#f39c12','#e74c3c','#8e44ad','#2980b9','#c0392b']
bars   = ax.bar(CFG['class_names'], per_class_f1, color=colors)
ax.set_ylim(0, 1); ax.set_ylabel('F1')
ax.set_title('Fusion Model — Per-Class F1 (Test Set)')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
for bar, v in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('/kaggle/working/fusion_per_class_f1.png', dpi=150)
plt.show()
print(f'Text  model (BERT)       test acc : 0.7282 ')
print(f'Image model (ResNet50)   test acc : 0.6780 ')
print(f' Fusion (prev, no weights) macro-F1: 0.2034 ')
print(f' Fusion (balanced weights) test acc: {test_acc:.4f}')
print(f'Fusion (balanced weights) macro-F1: {test_f1:.4f} ')
print('Outputs:')
for f in sorted(os.listdir('/kaggle/working')):
    if not f.endswith('.zip'):
        size = os.path.getsize(f'/kaggle/working/{f}') / 1e6
        print(f'  {f:<45} {size:.1f} MB')

shutil.make_archive('/kaggle/working/fusion_outputs', 'zip', '/kaggle/working')
print('\n✓ fusion_outputs.zip ready — download from Output tab')
