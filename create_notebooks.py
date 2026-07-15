import nbformat as nbf
import json


def make_training_notebook(model_name, model_code_func):
    """Create a complete training notebook for a given model."""
    nb = nbf.v4.new_notebook()
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {'name': 'python', 'version': '3.10.0'},
        'accelerator': 'GPU'
    }
    cells = []

    # =========================================================================
    # CELL 0: Markdown - Title
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(f"""# Treinamento {model_name} — Imagens Full (BUSBRA)

**Plano Ultra-Top-Master 3.0**

| Item | Valor |
|------|-------|
| Arquitetura | {model_name} (ImageNet pretrained) |
| Dataset | BUSBRA — imagens full (`processed/full`) |
| Validação | 5-fold Stratified**Group**KFold (split por paciente) |
| Seeds | 42, 123, 2024 |
| Métrica primária | AUC-ROC com IC 95% |
| Métricas secundárias | Sensibilidade, Especificidade, F1, Balanced Accuracy |
| Loss | BCEWithLogitsLoss com class weights |
| Early stopping | Patience = 15 épocas (monitorando AUC val) |
| Augmentation | **Nenhuma** (conforme decisão da pesquisadora) |
| Teste externo | BUSI (notebook separado, Semana 4) |

> Os resultados são salvos **incrementalmente** em JSON e CSV no Google Drive após cada fold. Se o Colab crashar, você não perde o que já treinou."""))

    # =========================================================================
    # CELL 1: Drive Mount
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 1. MONTAR GOOGLE DRIVE E DEFINIR CAMINHOS
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os

# ---- EDITE AQUI se necessário ----
BASE_DIR = '/content/drive/MyDrive/breast_ultrasound_anomalies'
# ----------------------------------

DATASET_CSV  = os.path.join(BASE_DIR, 'BUSBRA', 'BUSBRA', 'bus_data.csv')
IMAGES_DIR   = os.path.join(BASE_DIR, 'processed', 'full')
""" + f"RESULTS_DIR  = os.path.join(BASE_DIR, 'results', '{model_name}_full')" + """

os.makedirs(RESULTS_DIR, exist_ok=True)

# Verificação rápida
assert os.path.exists(DATASET_CSV), f"CSV não encontrado: {DATASET_CSV}"
assert os.path.isdir(IMAGES_DIR),   f"Diretório de imagens não encontrado: {IMAGES_DIR}"
print(f"✅ CSV: {DATASET_CSV}")
print(f"✅ Imagens: {IMAGES_DIR}")
print(f"✅ Resultados serão salvos em: {RESULTS_DIR}")"""))

    # =========================================================================
    # CELL 2: Imports
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 2. IMPORTS
# ==============================================================================
!pip install -q timm

import os, json, random, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')"""))

    # =========================================================================
    # CELL 3: Configurações
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 3. CONFIGURAÇÕES
# ==============================================================================
SEEDS     = [42, 123, 2024]
N_SPLITS  = 5
BATCH_SIZE = 32
EPOCHS     = 100
PATIENCE   = 15
LR         = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE   = 224

def seed_everything(seed):
    \"\"\"Fixa todas as fontes de aleatoriedade para reprodutibilidade.\"\"\"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""))

    # =========================================================================
    # CELL 4: Dataset + Transforms
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 4. DATASET E TRANSFORMS (sem augmentation)
# ==============================================================================
class BUSBRADataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ID']

        # Tentar .png primeiro (é o que existe em processed/full)
        img_path = os.path.join(self.img_dir, img_id + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_id + '.jpg')

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.float32)
        return image, label

# Normalização padrão ImageNet (usada por ResNet e Swin pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Sem augmentation — apenas resize + normalização
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])"""))

    # =========================================================================
    # CELL 5: Model
    # =========================================================================
    cells.append(nbf.v4.new_code_cell(model_code_func))

    # =========================================================================
    # CELL 6: Train/Eval functions
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 6. FUNÇÕES DE TREINO E AVALIAÇÃO
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc  = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_auc


def evaluate(model, loader, criterion):
    \"\"\"Avalia o modelo e retorna dicionário com todas as métricas.\"\"\"
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_bin  = (all_probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin, labels=[0,1]).ravel()

    metrics = {
        'loss':              running_loss / len(loader.dataset),
        'auc':               roc_auc_score(all_labels, all_probs),
        'accuracy':          accuracy_score(all_labels, preds_bin),
        'balanced_accuracy': balanced_accuracy_score(all_labels, preds_bin),
        'f1':                f1_score(all_labels, preds_bin, zero_division=0),
        'sensitivity':       tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity':       tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    return metrics, all_probs, all_labels"""))

    # =========================================================================
    # CELL 7: Main training loop
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 7. LOOP PRINCIPAL: 3 Seeds × 5 Folds
# ==============================================================================

# Carregar metadados
df = pd.read_csv(DATASET_CSV)
df = df.dropna(subset=['Pathology'])
df['label'] = df['Pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

# Filtrar apenas imagens que existem em processed/full
existing = []
for img_id in df['ID']:
    if os.path.exists(os.path.join(IMAGES_DIR, img_id + '.png')) or \\
       os.path.exists(os.path.join(IMAGES_DIR, img_id + '.jpg')):
        existing.append(img_id)
df = df[df['ID'].isin(existing)].reset_index(drop=True)

n_benign    = (df['label'] == 0).sum()
n_malignant = (df['label'] == 1).sum()
pos_weight  = torch.tensor([n_benign / n_malignant]).to(device)

print(f"Dataset: {len(df)} imagens ({n_benign} benignas, {n_malignant} malignas)")
print(f"Pacientes únicos (Case): {df['Case'].nunique()}")
print(f"pos_weight (class weight para malignos): {pos_weight.item():.4f}")
print(f"Configuração: {len(SEEDS)} seeds × {N_SPLITS} folds = {len(SEEDS)*N_SPLITS} runs")

# Arquivo JSON para salvar incrementalmente
results_json_path = os.path.join(RESULTS_DIR, 'all_fold_results.json')

# Carregar resultados anteriores se existirem (para retomar treinamento interrompido)
if os.path.exists(results_json_path):
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
    completed = {(r['seed'], r['fold']) for r in all_results}
    print(f"\\n⚠️  Encontrados {len(all_results)} runs anteriores. Retomando de onde parou.")
else:
    all_results = []
    completed = set()

# Histórico de treinamento (loss/auc por época) para gráficos
all_histories = []

for seed in SEEDS:
    seed_everything(seed)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], df['Case'])):

        # Pular folds já completados
        if (seed, fold) in completed:
            print(f"\\nSeed {seed} | Fold {fold+1}/{N_SPLITS} — JÁ COMPLETADO, pulando.")
            continue

        print(f"\\n{'='*60}")
        print(f"Seed {seed} | Fold {fold+1}/{N_SPLITS}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        print(f"  Train: {len(train_df)} imgs ({(train_df['label']==1).sum()} mal)")
        print(f"  Val:   {len(val_df)} imgs ({(val_df['label']==1).sum()} mal)")

        # Verificar vazamento por paciente
        train_cases = set(train_df['Case'])
        val_cases   = set(val_df['Case'])
        leak = train_cases & val_cases
        assert len(leak) == 0, f"VAZAMENTO DE DADOS! Pacientes em ambos: {leak}"
        print(f"  ✅ Sem vazamento de dados ({len(train_cases)} + {len(val_cases)} pacientes)")

        train_loader = DataLoader(
            BUSBRADataset(train_df, IMAGES_DIR, data_transform),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            BUSBRADataset(val_df, IMAGES_DIR, data_transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        model = get_model().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_auc = 0.0
        best_state = None
        best_metrics = {}
        patience_counter = 0
        history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

        for epoch in range(EPOCHS):
            t_loss, t_auc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_metrics, _, _ = evaluate(model, val_loader, criterion)

            history['train_loss'].append(t_loss)
            history['train_auc'].append(t_auc)
            history['val_loss'].append(v_metrics['loss'])
            history['val_auc'].append(v_metrics['auc'])

            improved = '⬆️' if v_metrics['auc'] > best_auc else ''
            print(f"  Ep {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss {t_loss:.4f} AUC {t_auc:.4f} | "
                  f"Val Loss {v_metrics['loss']:.4f} AUC {v_metrics['auc']:.4f} {improved}")

            if v_metrics['auc'] > best_auc:
                best_auc = v_metrics['auc']
                best_metrics = v_metrics
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stopping na época {epoch+1} (patience={PATIENCE})")
                break

        # Salvar modelo
        model_path = os.path.join(RESULTS_DIR, f'model_seed{seed}_fold{fold}.pth')
        torch.save(best_state, model_path)
        print(f"  💾 Modelo salvo: {model_path}")

        # Registrar resultado
        result = {
            'seed': seed,
            'fold': fold,
            'best_epoch': epoch + 1 - patience_counter,
            'total_epochs': epoch + 1,
            **best_metrics
        }
        all_results.append(result)

        # Salvar incrementalmente no JSON (no Drive!)
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Salvar histórico
        all_histories.append({
            'seed': seed, 'fold': fold, 'history': history
        })

        print(f"  📊 Melhor AUC: {best_auc:.4f} | Resultados salvos ({len(all_results)}/{len(SEEDS)*N_SPLITS} runs)")

print(f"\\n{'='*60}")
print(f"✅ TREINAMENTO COMPLETO! {len(all_results)} runs finalizados.")
print(f"{'='*60}")"""))

    # =========================================================================
    # CELL 8: Results Table
    # =========================================================================
    cells.append(nbf.v4.new_code_cell(f"""# ==============================================================================
# 8. TABELA DE RESULTADOS
# ==============================================================================

# Carregar resultados do JSON (funciona mesmo se o kernel reiniciou)
results_json_path = os.path.join(RESULTS_DIR, 'all_fold_results.json')
with open(results_json_path, 'r') as f:
    all_results = json.load(f)

df_results = pd.DataFrame(all_results)

# Salvar também como CSV
csv_path = os.path.join(RESULTS_DIR, 'resultados_{model_name}.csv')
df_results.to_csv(csv_path, index=False)
print(f"CSV salvo em: {{csv_path}}")

# Exibir tabela completa
print("\\n📋 Resultados por Seed × Fold:")
print("=" * 100)
display_cols = ['seed', 'fold', 'best_epoch', 'auc', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy']
print(df_results[display_cols].to_string(index=False, float_format='%.4f'))

# Resumo por seed
print("\\n📋 Resumo por Seed:")
print("=" * 60)
for seed in SEEDS:
    seed_data = df_results[df_results['seed'] == seed]
    print(f"  Seed {{seed}}: AUC = {{seed_data['auc'].mean():.4f}} ± {{seed_data['auc'].std():.4f}}")

# Resumo geral
print("\\n📋 RESUMO GERAL:")
print("=" * 60)
metrics_to_report = ['auc', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'accuracy']
for m in metrics_to_report:
    vals = df_results[m]
    print(f"  {{m.upper():25s}}: {{vals.mean():.4f}} ± {{vals.std():.4f}}")"""))

    # =========================================================================
    # CELL 9: Visualizations
    # =========================================================================
    cells.append(nbf.v4.new_code_cell(f"""# ==============================================================================
# 9. VISUALIZAÇÕES
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('{model_name} — Resultados 5-Fold CV × 3 Seeds', fontsize=14, fontweight='bold')

# 9a. Boxplot de AUC por Seed
ax = axes[0]
seed_aucs = [df_results[df_results['seed'] == s]['auc'].values for s in SEEDS]
bp = ax.boxplot(seed_aucs, labels=[str(s) for s in SEEDS], patch_artist=True)
colors = ['#3498db', '#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xlabel('Seed')
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC por Seed')
ax.axhline(y=df_results['auc'].mean(), color='gray', linestyle='--', alpha=0.5, label=f'Média: {{df_results["auc"].mean():.4f}}')
ax.legend()

# 9b. Barplot de todas as métricas
ax = axes[1]
metrics_names = ['AUC', 'F1', 'Sens.', 'Spec.', 'Bal.Acc.']
metrics_keys  = ['auc', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy']
means = [df_results[k].mean() for k in metrics_keys]
stds  = [df_results[k].std() for k in metrics_keys]
bars = ax.bar(metrics_names, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'], alpha=0.7)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Valor')
ax.set_title('Métricas (média ± std)')
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{{mean:.3f}}', ha='center', va='bottom', fontsize=9)

# 9c. Convergência (último fold treinado — se houver histórico disponível)
ax = axes[2]
if len(all_histories) > 0:
    last_h = all_histories[-1]['history']
    epochs_range = range(1, len(last_h['train_loss']) + 1)
    ax.plot(epochs_range, last_h['train_auc'], label='Train AUC', color='#3498db')
    ax.plot(epochs_range, last_h['val_auc'], label='Val AUC', color='#e74c3c')
    ax.set_xlabel('Época')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(f'Convergência (Seed {{all_histories[-1]["seed"]}}, Fold {{all_histories[-1]["fold"]+1}})')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Histórico não disponível\\n(kernel reiniciou?)', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Convergência')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'resultados_visuais.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico salvo em: {{os.path.join(RESULTS_DIR, 'resultados_visuais.png')}}")"""))

    # =========================================================================
    # CELL 10: IC 95% Bootstrap
    # =========================================================================
    cells.append(nbf.v4.new_code_cell(f"""# ==============================================================================
# 10. INTERVALO DE CONFIANÇA 95% (Bootstrap)
# ==============================================================================

def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    \"\"\"Calcula IC via bootstrap percentile.\"\"\"
    rng = np.random.RandomState(seed)
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper

print("📊 RESUMO FINAL COM IC 95% (Bootstrap)")
print("=" * 70)
metrics_report = [
    ('AUC-ROC (primária)',  'auc'),
    ('F1',                  'f1'),
    ('Sensibilidade',       'sensitivity'),
    ('Especificidade',      'specificity'),
    ('Balanced Accuracy',   'balanced_accuracy'),
    ('Accuracy',            'accuracy'),
]

summary_rows = []
for name, key in metrics_report:
    vals = df_results[key].values
    mean = vals.mean()
    std  = vals.std()
    lo, hi = bootstrap_ci(vals)
    print(f"  {{name:25s}}: {{mean:.4f}} ± {{std:.4f}}  [IC 95%: {{lo:.4f}} — {{hi:.4f}}]")
    summary_rows.append({{
        'Métrica': name, 'Média': mean, 'Std': std,
        'IC95_lower': lo, 'IC95_upper': hi
    }})

# Salvar resumo
df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(RESULTS_DIR, 'resumo_{model_name}.csv')
df_summary.to_csv(summary_path, index=False)
print(f"\\n💾 Resumo salvo em: {{summary_path}}")"""))

    nb['cells'] = cells
    return nb


# =============================================================================
# DEFINIÇÃO DOS MODELOS
# =============================================================================

RESNET50_CODE = """# ==============================================================================
# 5. MODELO: ResNet-50 (ImageNet pretrained)
# ==============================================================================
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Saída binária
    return model

print(f"Parâmetros treináveis: {sum(p.numel() for p in get_model().parameters() if p.requires_grad):,}")"""

SWIN_CODE = """# ==============================================================================
# 5. MODELO: Swin-T (ImageNet pretrained via timm)
# ==============================================================================
def get_model():
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=1)
    return model

print(f"Parâmetros treináveis: {sum(p.numel() for p in get_model().parameters() if p.requires_grad):,}")"""


# =============================================================================
# GERAR NOTEBOOKS
# =============================================================================
print("Gerando ResNet50_Training.ipynb...")
nb_resnet = make_training_notebook("ResNet50", RESNET50_CODE)
with open('ResNet50_Training.ipynb', 'w') as f:
    nbf.write(nb_resnet, f)

print("Gerando SwinViT_Training.ipynb...")
nb_swin = make_training_notebook("SwinViT", SWIN_CODE)
with open('SwinViT_Training.ipynb', 'w') as f:
    nbf.write(nb_swin, f)

print("✅ Notebooks de treinamento gerados!")
