import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T
import torchvision.models as models
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
CFG = {

    'train_tsv':  '/kaggle/input/datasets/yashmogha/dataset/multimodal_train_reduced.tsv',
    'val_tsv':    '/kaggle/input/datasets/yashmogha/dataset/multimodal_validate.tsv',
    'test_tsv':   '/kaggle/input/datasets/yashmogha/dataset/multimodal_test_public.tsv',
    'train_dir':  '/kaggle/input/datasets/yashmogha/dataset/Archive/downloaded',
    'val_dir':    '/kaggle/input/datasets/yashmogha/dataset/Archive/downloaded_images',
    'test_dir':   '/kaggle/input/datasets/yashmogha/dataset/test',
    'out_dir':    '/kaggle/working',

    'num_labels':  6,
    'freeze_bn':   True,
    'dropout':     0.3,
    'image_size':  224,


    'epochs':       5,
    'batch_size':   32,
    'lr':           1e-4,      # head LR
    'backbone_lr':  1e-5,      # backbone LR — 10x lower
    'weight_decay': 1e-4,
    'num_workers':  2,
    'persistent_workers': True,
    'prefetch_factor':    2,
    'label_col':    '6_way_label',

    'class_names': [
        'true', 'satire/parody', 'misleading content',
        'imposter content', 'false connection', 'manipulated content',
    ],
}

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.webp']
CKPT     = os.path.join(CFG['out_dir'], 'image_best.pt')
print('Config loaded.')
print(f'Differential LR — backbone: {CFG["backbone_lr"]}  head: {CFG["lr"]}')
def find_image_path(image_dir, img_id):

    for ext in IMG_EXTS:
        path = os.path.join(image_dir, f'{img_id}{ext}')
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    img.convert('RGB')
                return path
            except Exception:
                pass
    return None


class ImageDataset(Dataset):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, tsv_path, image_dir, split='train',
                 image_size=224, label_col='6_way_label'):

        df = pd.read_csv(tsv_path, sep='\t', dtype=str)
        df = df[df[label_col].notna()].reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)


        before = len(df)
        df['_img_path'] = df['id'].apply(
            lambda img_id: find_image_path(image_dir, str(img_id).strip())
        )
        df   = df[df['_img_path'].notna()]

        df   = df.reset_index(drop=False)
        df   = df.rename(columns={'index': '_tsv_row'})
        after = len(df)
        print(f'  [{split}] Dropped {before-after:,}/{before:,} '
              f'({(before-after)/before*100:.1f}% missing/corrupt) '
              f'→ {after:,} samples')


        self.df        = df
        self.label_col = label_col

        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.Normalize(self.IMG_MEAN, self.IMG_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(self.IMG_MEAN, self.IMG_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['_img_path']).convert('RGB')
        return {
            'image':    self.transform(img),
            'label':    torch.tensor(int(row[self.label_col]), dtype=torch.long),
            'tsv_row':  torch.tensor(int(row['_tsv_row']),    dtype=torch.long),
        }
def build_loaders(cfg):
    shared = dict(image_size=cfg['image_size'], label_col=cfg['label_col'])
    nw     = cfg['num_workers']
    loader_kw = dict(
        batch_size         = cfg['batch_size'],
        num_workers        = nw,
        pin_memory         = True,
        persistent_workers = cfg['persistent_workers'] if nw > 0 else False,
        prefetch_factor    = cfg['prefetch_factor']    if nw > 0 else None,
    )

    print('Building datasets (validating images)...')
    train_ds = ImageDataset(cfg['train_tsv'], cfg['train_dir'], split='train', **shared)
    val_ds   = ImageDataset(cfg['val_tsv'],   cfg['val_dir'],   split='val',   **shared)
    test_ds  = ImageDataset(cfg['test_tsv'],  cfg['test_dir'],  split='test',  **shared)

    # Recompute class weights from filtered train set
    filtered_counts = torch.zeros(cfg['num_labels'])
    for lbl in train_ds.df[cfg['label_col']].values:
        filtered_counts[lbl] += 1
    print(f'Filtered train class counts: {filtered_counts.int().tolist()}')

    class_weights  = 1.0 / torch.sqrt(filtered_counts.clamp(min=1))
    class_weights  = class_weights / class_weights.mean()
    sample_weights = class_weights[train_ds.df[cfg['label_col']].values]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_dl = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False,   **loader_kw)
    test_dl  = DataLoader(test_ds,  shuffle=False,   **loader_kw)

    print(f'Sizes → Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')
    print(f'Class weights (normalized): {class_weights.numpy().round(3)}')
    return train_dl, val_dl, test_dl, class_weights, train_ds, val_ds, test_ds


train_dl, val_dl, test_dl, class_weights, train_ds, val_ds, test_ds = build_loaders(CFG)

MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

samples = {}
for batch in train_dl:
    for img, lbl in zip(batch['image'], batch['label']):
        l = lbl.item()
        if l not in samples: samples[l] = img
        if len(samples) == CFG['num_labels']: break
    if len(samples) == CFG['num_labels']: break

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for i in range(6):
    vis = (samples[i] * STD_T + MEAN_T).clamp(0,1).permute(1,2,0).numpy()
    axes[i].imshow(vis); axes[i].axis('off')
    axes[i].set_title(CFG['class_names'][i], fontsize=8)
plt.suptitle('One real image per class — no grey placeholders')
plt.tight_layout()
plt.savefig('/kaggle/working/sample_images.png', dpi=150)
plt.show()
class ImageModel(nn.Module):

    def __init__(self, num_labels=6, dropout=0.3, freeze_bn=True):
        super().__init__()
        backbone       = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.embed_dim = backbone.fc.in_features   # 2048
        self.encoder   = nn.Sequential(*list(backbone.children())[:-1])
        self.drop      = nn.Dropout(dropout)
        self.head      = nn.Linear(self.embed_dim, num_labels)

        if freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def get_embedding(self, image):
        return self.encoder(image).flatten(1)   # (B, 2048)

    def forward(self, image):
        return self.head(self.drop(self.get_embedding(image)))

    def get_param_groups(self, backbone_lr, head_lr):
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params     = list(self.drop.parameters()) + list(self.head.parameters())
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': head_lr},
        ]


model = ImageModel(
    num_labels = CFG['num_labels'],
    dropout    = CFG['dropout'],
    freeze_bn  = CFG['freeze_bn'],
).to(DEVICE)



total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params — total: {total/1e6:.1f}M | trainable: {trainable/1e6:.1f}M')
print(f'Differential LR — backbone: {CFG["backbone_lr"]}  head: {CFG["lr"]}')
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for batch in tqdm(loader, desc='train', leave=False):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=DEVICE.type == 'cuda'):
            logits = model(images)
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
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        logits = model(images); loss = criterion(logits, labels); preds = logits.argmax(1)
        loss_sum += loss.item() * labels.size(0)
        correct  += (preds == labels).sum().item(); n += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return loss_sum / n, correct / n, all_preds, all_labels
param_groups = model.get_param_groups(
    backbone_lr = CFG['backbone_lr'],
    head_lr     = CFG['lr'],
)
optimizer    = AdamW(param_groups, weight_decay=CFG['weight_decay'])
total_steps  = CFG['epochs'] * len(train_dl)
warmup_steps = total_steps // 10
scheduler    = get_cosine_schedule_with_warmup(optimizer,
                   num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion    = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
scaler       = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

print(f'Total steps: {total_steps} | Warmup: {warmup_steps}')
print(f'Loss weights: {class_weights.numpy().round(3)}')

best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, CFG['epochs'] + 1):
    tl, ta          = train_epoch(model, train_dl, optimizer, criterion, scaler, scheduler)
    vl, va, vp, vl_ = eval_epoch(model, val_dl, criterion)

    history['train_loss'].append(tl); history['train_acc'].append(ta)
    history['val_loss'].append(vl);   history['val_acc'].append(va)

    flag = ''
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), CKPT)
        flag = '  ✓ saved'

    print(f'Epoch {epoch:02d}/{CFG["epochs"]}  '
          f'train {tl:.4f}/{ta:.4f}  '
          f'val {vl:.4f}/{va:.4f}  '
          f'lr_head {scheduler.get_last_lr()[-1]:.2e}{flag}')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
eps = range(1, CFG['epochs'] + 1)
axes[0].plot(eps, history['train_loss'], label='train')
axes[0].plot(eps, history['val_loss'],   label='val')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(eps, history['train_acc'], label='train')
axes[1].plot(eps, history['val_acc'],   label='val')
axes[1].set_title('Accuracy'); axes[1].legend()
plt.suptitle('Image Model — Training History')
plt.tight_layout()
plt.savefig('/kaggle/working/image_history.png', dpi=150)
plt.show()
def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm      = confusion_matrix(labels, preds).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, t in zip(axes, [cm.astype(int), cm_norm],
                                ['d', '.2f'], ['Raw Counts', 'Normalised']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{title} — {t}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.show()


def full_eval(model, loader, criterion, split_name, save_prefix):
    loss, acc, preds, labels = eval_epoch(model, loader, criterion, desc=split_name)
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f'\n{"="*55}')
    print(f'{split_name.upper()} SET  —  loss: {loss:.4f}  acc: {acc:.4f}  macro-F1: {macro_f1:.4f}')
    print(f'{"="*55}\n')
    print(classification_report(labels, preds, target_names=CFG['class_names'], digits=4))
    plot_confusion_matrix(labels, preds, CFG['class_names'],
                          f'Image Model ({split_name})',
                          f'/kaggle/working/{save_prefix}_cm.png')
    return loss, acc
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
criterion_eval = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
print(f'Loaded best checkpoint: {CKPT}')
val_loss, val_acc = full_eval(model, val_dl,  criterion_eval, 'Validation', 'image_val')
test_loss, test_acc = full_eval(model, test_dl, criterion_eval, 'Test', 'image_test')
print('SUMMARY')
print(f'  Best val acc (training) : {best_val_acc:.4f}')
print(f'  Val  acc (best ckpt)    : {val_acc:.4f}')
print(f'  Test acc (best ckpt)    : {test_acc:.4f}')
@torch.no_grad()
def extract_embeddings(model, dataset, split_name, cfg):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    all_embeds, all_labels, all_rows = [], [], []
    for batch in tqdm(loader, desc=f'Extracting [{split_name}]'):
        all_embeds.append(model.get_embedding(batch['image'].to(DEVICE)).cpu())
        all_labels.append(batch['label'])
        all_rows.append(batch['tsv_row'])

    embeds      = torch.cat(all_embeds)
    labels      = torch.cat(all_labels)
    row_indices = torch.cat(all_rows)
    save_path = os.path.join(cfg['out_dir'], f'image_{split_name}_embeds.pt')
    torch.save({
        'embeds':      embeds,
        'labels':      labels,
        'row_indices': row_indices,
    }, save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f'  Saved {save_path}')
    print(f'    embeds: {tuple(embeds.shape)}  '
          f'labels: {tuple(labels.shape)}  '
          f'row_indices: {tuple(row_indices.shape)}  '
          f'({size_mb:.1f} MB)')


model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
print('Extracting embeddings (shuffle=False, row_indices saved)...')

for split_name, dataset in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
    extract_embeddings(model, dataset, split_name, CFG)

for split_name in ['train', 'val', 'test']:
    data = torch.load(f'/kaggle/working/image_{split_name}_embeds.pt')
    e, l, r = data['embeds'], data['labels'], data['row_indices']
    print(f'  {split_name:<6}  embeds: {tuple(e.shape)}  '
          f'labels: {tuple(l.shape)}  '
          f'row_indices: {tuple(r.shape)}  '
          f'rows [{r.min().item()}–{r.max().item()}]')

print('\n✓ Verified — row_indices saved, fusion alignment ready.')
print('\nFiles:')
for f in sorted(os.listdir('/kaggle/working')):
    size = os.path.getsize(f'/kaggle/working/{f}') / 1e6
    print(f'  {f:<45} {size:.1f} MB')
import shutil
shutil.make_archive('/kaggle/working/outputs', 'zip', '/kaggle/working')
print('✓ outputs.zip ready — download from Output tab')