
import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

CFG = {
    'train_tsv':  '/kaggle/input/fakeddit/multimodal_train.tsv',
    'val_tsv':    '/kaggle/input/fakeddit/multimodal_validate.tsv',
    'test_tsv':   '/kaggle/input/fakeddit/multimodal_test_public.tsv',
    'out_dir':    '/kaggle/working',


    'model_name':    'bert-base-uncased',
    'num_labels':    6,
    'freeze_layers': 8,
    'dropout':       0.3,


    'max_text_len':  128,
    'label_col':     '6_way_label',
    'text_col':      'clean_title',


    'epochs':        10,
    'batch_size':    64,
    'lr':            2e-5,
    'weight_decay':  1e-4,
    'num_workers':   2,

    'class_names': [
        'true',
        'satire/parody',
        'misleading content',
        'imposter content',
        'false connection',
        'manipulated content',
    ],
}

CKPT = os.path.join(CFG['out_dir'], 'text_best.pt')


CLASS_COUNTS = [88832, 33481, 42888, 11784, 67143, 21576]
print('Config loaded.')
class TextDataset(Dataset):
    def __init__(self, tsv_path, split='train',
                 tokenizer_name='bert-base-uncased',
                 max_len=128, label_col='6_way_label', text_col='clean_title'):

        df = pd.read_csv(tsv_path, sep='\t', dtype=str)
        df = df[df[label_col].notna()].reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)
        self.df        = df
        self.label_col = label_col
        self.text_col  = text_col if text_col in df.columns else 'title'
        self.max_len   = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row[self.text_col])
        label = int(row[self.label_col])

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
        }


def build_loaders(cfg):
    shared = dict(
        tokenizer_name = cfg['model_name'],
        max_len        = cfg['max_text_len'],
        label_col      = cfg['label_col'],
        text_col       = cfg['text_col'],
    )
    loader_kw = dict(
        batch_size  = cfg['batch_size'],
        num_workers = cfg['num_workers'],
        pin_memory  = True,
    )

    train_ds = TextDataset(cfg['train_tsv'], split='train', **shared)
    val_ds   = TextDataset(cfg['val_tsv'],   split='val',   **shared)
    test_ds  = TextDataset(cfg['test_tsv'],  split='test',  **shared)

    class_counts  = torch.tensor(CLASS_COUNTS, dtype=torch.float)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_ds.df[cfg['label_col']].values]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_dl = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False,   **loader_kw)
    test_dl  = DataLoader(test_ds,  shuffle=False,   **loader_kw)

    print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')
    print(f'WeightedRandomSampler enabled — class weights: {class_weights.numpy().round(6)}')
    return train_dl, val_dl, test_dl, class_weights


train_dl, val_dl, test_dl, class_weights = build_loaders(CFG)

df_train = pd.read_csv(CFG['train_tsv'], sep='\t', dtype=str)
df_train = df_train[df_train[CFG['label_col']].notna()]
counts   = df_train[CFG['label_col']].astype(int).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))


axes[0].bar(CFG['class_names'], counts.values, color='steelblue')
axes[0].set_title('Class Distribution (raw counts)')
axes[0].set_ylabel('Count')
plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=8)


total      = sum(counts.values)
ideal      = 100 / len(counts)
actual_pct = [c / total * 100 for c in counts.values]
axes[1].bar(CFG['class_names'], actual_pct, color='salmon', label='Actual %')
axes[1].axhline(ideal, color='green', linestyle='--', label=f'Ideal ({ideal:.1f}%)')
axes[1].set_title('Class Distribution (% vs ideal)')
axes[1].set_ylabel('Percentage')
axes[1].legend()
plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')

plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png', dpi=150)
plt.show()

max_imbalance = max(counts.values) / min(counts.values)
print(f'Imbalance ratio (max/min): {max_imbalance:.1f}x')
print(counts.rename(index=dict(enumerate(CFG['class_names']))).to_string())
class TextModel(nn.Module):

    def __init__(self, num_labels=6, model_name='bert-base-uncased',
                 dropout=0.3, freeze_layers=8):
        super().__init__()
        self.bert      = BertModel.from_pretrained(model_name)
        self.embed_dim = self.bert.config.hidden_size
        self.drop      = nn.Dropout(dropout)
        self.head      = nn.Linear(self.embed_dim, num_labels)
        self._freeze(freeze_layers)

    def _freeze(self, n):
        if n <= 0:
            return
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def get_embedding(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask):
        return self.head(self.drop(self.get_embedding(input_ids, attention_mask)))


model = TextModel(
    num_labels    = CFG['num_labels'],
    model_name    = CFG['model_name'],
    dropout       = CFG['dropout'],
    freeze_layers = CFG['freeze_layers'],
).to(DEVICE)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params — total: {total/1e6:.1f}M | trainable: {trainable/1e6:.1f}M')
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for batch in tqdm(loader, desc='train', leave=False):
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=DEVICE.type == 'cuda'):
            logits = model(ids, mask)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * labels.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        n        += labels.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, desc='val'):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        loss_sum += loss.item() * labels.size(0)
        correct  += (preds == labels).sum().item()
        n        += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return loss_sum / n, correct / n, all_preds, all_labels
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG['lr'], weight_decay=CFG['weight_decay']
)
scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
print(f'Loss weights: {class_weights.numpy().round(4)}')

scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, CFG['epochs'] + 1):
    tl, ta          = train_epoch(model, train_dl, optimizer, criterion, scaler)
    vl, va, vp, vl_ = eval_epoch(model, val_dl, criterion)
    scheduler.step()

    history['train_loss'].append(tl); history['train_acc'].append(ta)
    history['val_loss'].append(vl);   history['val_acc'].append(va)

    flag = ''
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), CKPT)
        flag = '  ✓ saved'

    print(f'Epoch {epoch:02d}/{CFG["epochs"]}  '
          f'train {tl:.4f}/{ta:.4f}  '
          f'val {vl:.4f}/{va:.4f}{flag}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
eps = range(1, CFG['epochs'] + 1)

axes[0].plot(eps, history['train_loss'], label='train')
axes[0].plot(eps, history['val_loss'],   label='val')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(eps, history['train_acc'], label='train')
axes[1].plot(eps, history['val_acc'],   label='val')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()

plt.suptitle('Text Model — Training History', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/text_history.png', dpi=150)
plt.show()

def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm.astype(int), annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
    )
    axes[0].set_title(f'{title} — Raw Counts')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right')


    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
    )
    axes[1].set_title(f'{title} — Normalised')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def full_eval(model, loader, criterion, split_name, save_prefix):
    loss, acc, preds, labels = eval_epoch(model, loader, criterion, desc=split_name)
    print(f'\n{"="*55}')
    print(f'{split_name.upper()} SET  —  loss: {loss:.4f}  |  acc: {acc:.4f}')
    print(f'{"="*55}\n')
    print(classification_report(labels, preds,
                                 target_names=CFG['class_names'], digits=4))
    plot_confusion_matrix(
        labels, preds, CFG['class_names'],
        title=f'Text Model ({split_name})',
        save_path=f'/kaggle/working/{save_prefix}_cm.png',
    )
    return loss, acc, preds, labels

model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
criterion_eval = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
print(f'Loaded best checkpoint from: {CKPT}')

val_loss, val_acc, val_preds, val_labels = full_eval(
    model, val_dl, criterion_eval,
    split_name='Validation', save_prefix='text_val'
)

test_loss, test_acc, test_preds, test_labels = full_eval(
    model, test_dl, criterion_eval,
    split_name='Test', save_prefix='text_test'
)
print('SUMMARY')
print(f'  Best val acc (during training) : {best_val_acc:.4f}')
print(f'  Val  acc (best ckpt)           : {val_acc:.4f}')
print(f'  Test acc (best ckpt)           : {test_acc:.4f}')
print(f'\nCheckpoint : {CKPT}')
print('\nOutputs in /kaggle/working/')
for f in sorted(os.listdir('/kaggle/working')):
    size = os.path.getsize(f'/kaggle/working/{f}') / 1e6
    print(f'  {f:<35} {size:.1f} MB')


    @torch.no_grad()
    def extract_embeddings(model, loader, split_name):
        model.eval()
        all_embeds, all_labels = [], []

        for batch in tqdm(loader, desc=f'Extracting [{split_name}]'):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label']

            embeds = model.get_embedding(ids, mask)
            all_embeds.append(embeds.cpu())
            all_labels.append(labels)

        return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    splits = [
        ('train', train_dl),
        ('val', val_dl),
        ('test', test_dl),
    ]

    for split_name, loader in splits:
        embeds, labels = extract_embeddings(model, loader, split_name)
        save_path = f'/kaggle/working/text_{split_name}_embeds.pt'
        torch.save({'embeds': embeds, 'labels': labels}, save_path)
        print(f'  Saved {save_path}  →  shape: {embeds.shape}  ({os.path.getsize(save_path) / 1e6:.1f} MB)')
# ── Verify saved embeddings ────────────────────────────────────────
for split_name in ['train', 'val', 'test']:
    data = torch.load(f'/kaggle/working/text_{split_name}_embeds.pt')
    e, l = data['embeds'], data['labels']
    print(f'  {split_name:<6}  embeds: {tuple(e.shape)}  labels: {tuple(l.shape)}  '
          f'dtype: {e.dtype}  label range: [{l.min().item()}, {l.max().item()}]')

print('\n✓ All embeddings verified.')
print('\nFiles ready for fusion notebook:')
print('  /kaggle/working/text_best.pt')
print('  /kaggle/working/text_train_embeds.pt')
print('  /kaggle/working/text_val_embeds.pt')
print('  /kaggle/working/text_test_embeds.pt')
print('\nIn the fusion notebook, load with:')
print("  data = torch.load('text_train_embeds.pt')")
print("  text_embeds = data['embeds']   # (N, 768)")
print("  labels      = data['labels']   # (N,)")