#!/usr/bin/env python3
"""
Gera o notebook YOLO26_Seg_Training.ipynb para treinamento no Google Colab.

A YOLO26n-seg faz segmentação de instâncias + classificação simultaneamente:
detecta o nódulo, segmenta-o e classifica como benigno ou maligno.

Uso:
    python create_yolo_notebook.py
"""

import nbformat as nbf


# =============================================================================
# CELL DEFINITIONS
# =============================================================================

MARKDOWN_TITLE = r'''# Treinamento YOLO26n-seg — Segmentação + Classificação (BUSBRA)

**Plano Ultra-Top-Master 3.0 — Versão YOLO26**

| Item | Valor |
|------|-------|
| Arquitetura | YOLO26n-seg (pretrained COCO) |
| Tarefa | Segmentação de instâncias + Classificação (benigno × maligno) |
| Dataset | BUSBRA — imagens originais + máscaras de segmentação |
| Validação | 5-fold StratifiedGroupKFold (split por paciente) |
| Seeds | 42, 123, 2024 |
| Métrica primária (classificação) | AUC-ROC com IC 95% |
| Métricas secundárias | Sensibilidade, Especificidade, F1, Balanced Accuracy |
| Métricas de segmentação | mAP50, mAP50-95 (box e mask) |
| Early stopping | Patience = 15 épocas (monitorando fitness) |
| Augmentation | **Nenhuma** (conforme decisão da pesquisadora) |
| Teste externo | BUSI (notebook separado, Semana 4) |

> A YOLO26 realiza **segmentação + classificação simultaneamente**: detecta o nódulo na imagem, segmenta-o e classifica como benigno ou maligno. Isso **elimina a dependência de segmentação prévia** do nódulo.

> O mecanismo de **self-attention** da YOLO26 busca captar os pontos mais importantes da imagem, complementando (sem substituir) o soft masking do pré-processamento.

> Os resultados são salvos **incrementalmente** em JSON no Google Drive após cada fold. Se o Colab crashar, você não perde o que já treinou.'''


CELL_1_DRIVE = r'''# ==============================================================================
# 1. MONTAR GOOGLE DRIVE E DEFINIR CAMINHOS
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os

# ---- EDITE AQUI se necessário ----
BASE_DIR = '/content/drive/MyDrive/breast_ultrasound_anomalies'
# ----------------------------------

DATASET_CSV = os.path.join(BASE_DIR, 'BUSBRA', 'BUSBRA', 'bus_data.csv')
IMAGES_DIR  = os.path.join(BASE_DIR, 'BUSBRA', 'BUSBRA', 'Images')
MASKS_DIR   = os.path.join(BASE_DIR, 'BUSBRA', 'BUSBRA', 'Masks')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'YOLO26_seg')
TEMP_DIR    = '/content/yolo_folds'  # Armazenamento local (mais rápido)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Verificação rápida
assert os.path.exists(DATASET_CSV), f"CSV não encontrado: {DATASET_CSV}"
assert os.path.isdir(IMAGES_DIR),   f"Diretório de imagens não encontrado: {IMAGES_DIR}"
assert os.path.isdir(MASKS_DIR),    f"Diretório de máscaras não encontrado: {MASKS_DIR}"
print(f"✅ CSV: {DATASET_CSV}")
print(f"✅ Imagens: {IMAGES_DIR}")
print(f"✅ Máscaras: {MASKS_DIR}")
print(f"✅ Resultados serão salvos em: {RESULTS_DIR}")'''


CELL_2_IMPORTS = r'''# ==============================================================================
# 2. IMPORTS
# ==============================================================================
!pip install -q -U ultralytics

import os, json, random, shutil, copy
import numpy as np
import pandas as pd
import cv2
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Dispositivo: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')'''


CELL_3_CONFIG = r'''# ==============================================================================
# 3. CONFIGURAÇÕES
# ==============================================================================
SEEDS      = [42, 123, 2024]
N_SPLITS   = 5
BATCH_SIZE = 8       # Menor que ResNet50 (segmentação é mais pesada em memória)
EPOCHS     = 150
PATIENCE   = 15
IMG_SIZE   = 512     # Resolução otimizada para ultrassom (~274×353 originais)

CLASS_NAMES = ['benign', 'malignant']

def seed_everything(seed):
    """Fixa todas as fontes de aleatoriedade para reprodutibilidade."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''


CELL_4_FUNCTIONS = r'''# ==============================================================================
# 4. FUNÇÕES: CONVERSÃO DE MÁSCARA + PREPARAÇÃO DE FOLD + MÉTRICAS
# ==============================================================================

def mask_to_yolo_polygons(mask_path):
    """Converte uma máscara binária PNG em polígonos normalizados (formato YOLO).

    Retorna lista de strings com coordenadas normalizadas: 'x1 y1 x2 y2 ... xN yN'
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    H, W = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        # Simplificar contorno para reduzir pontos (sem perder forma)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        xs = np.clip(pts[:, 0] / W, 0.0, 1.0)
        ys = np.clip(pts[:, 1] / H, 0.0, 1.0)
        coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in zip(xs, ys))
        polygons.append(coords)

    return polygons


def prepare_fold_dataset(train_df, val_df, fold_dir, images_dir, masks_dir):
    """Cria diretorios no formato YOLO para um fold especifico.

    Estrutura criada:
        fold_dir/
          images/
            train/
            val/
          labels/
            train/
            val/
          data.yaml
    """
    for split in ['train', 'val']:
        os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)

    stats = {
        'train': {'total': 0, 'with_mask': 0, 'skipped': 0},
        'val':   {'total': 0, 'with_mask': 0, 'skipped': 0}
    }

    for split, split_df in [('train', train_df), ('val', val_df)]:
        for _, row in split_df.iterrows():
            img_id = row['ID']
            label = row['label']  # 0=benigno, 1=maligno

            # Caminhos de origem
            img_src = os.path.join(images_dir, img_id + '.png')
            mask_name = img_id.replace('bus_', 'mask_') + '.png'
            mask_src = os.path.join(masks_dir, mask_name)

            if not os.path.exists(img_src):
                stats[split]['skipped'] += 1
                continue

            stats[split]['total'] += 1

            # Caminhos de destino
            img_dst = os.path.join(fold_dir, 'images', split, img_id + '.png')
            lbl_dst = os.path.join(fold_dir, 'labels', split, img_id + '.txt')

            # Symlink da imagem (economiza espaço e tempo)
            if os.path.exists(img_dst):
                os.remove(img_dst)
            os.symlink(img_src, img_dst)

            # Converter máscara para polígono YOLO
            if os.path.exists(mask_src):
                polygons = mask_to_yolo_polygons(mask_src)
                if polygons:
                    lines = [f'{label} {poly}' for poly in polygons]
                    with open(lbl_dst, 'w') as f:
                        f.write('\n'.join(lines))
                    stats[split]['with_mask'] += 1
                else:
                    with open(lbl_dst, 'w') as f:
                        f.write('')
            else:
                with open(lbl_dst, 'w') as f:
                    f.write('')

    # Gerar data.yaml
    data_yaml = {
        'path': fold_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': CLASS_NAMES
    }
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    return yaml_path, stats


def extract_classification_metrics(model, val_img_dir, gt_lookup):
    """Extrai metricas de classificacao (nivel de imagem) das predicoes YOLO.

    Para cada imagem:
    - Se detectou nodulo: classe predita = classe da deteccao mais confiante
      P(maligno) = conf se classe=1, ou 1-conf se classe=0
    - Se nao detectou nenhum nodulo: P(maligno) = 0.5 (incerteza maxima)

    Retorna: dict com metricas, array de probabilidades, array de labels
    """
    val_img_paths = sorted(Path(val_img_dir).glob('*.png'))

    all_probs = []
    all_labels = []
    n_no_detection = 0

    for img_path in tqdm(val_img_paths, desc='  Classificando', leave=False):
        img_id = img_path.stem
        if img_id not in gt_lookup:
            continue

        gt_label = gt_lookup[img_id]

        # Inferência
        results = model.predict(
            source=str(img_path),
            conf=0.1,
            iou=0.5,
            save=False,
            verbose=False
        )
        result = results[0]

        if len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            best_idx = np.argmax(confs)
            best_class = classes[best_idx]
            best_conf = float(confs[best_idx])

            if best_class == 1:  # maligno
                prob_malignant = best_conf
            else:  # benigno
                prob_malignant = 1.0 - best_conf
        else:
            prob_malignant = 0.5
            n_no_detection += 1

        all_probs.append(prob_malignant)
        all_labels.append(gt_label)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_bin = (all_probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        'auc':               auc,
        'accuracy':          accuracy_score(all_labels, preds_bin),
        'balanced_accuracy': balanced_accuracy_score(all_labels, preds_bin),
        'f1':                f1_score(all_labels, preds_bin, zero_division=0),
        'sensitivity':       tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity':       tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'n_no_detection':    int(n_no_detection),
    }

    return metrics, all_probs, all_labels'''


CELL_5_LOAD_CSV = r'''# ==============================================================================
# 5. CARREGAR DATASET E PREPARAR METADADOS
# ==============================================================================

df = pd.read_csv(DATASET_CSV)
df = df.dropna(subset=['Pathology'])
df['label'] = df['Pathology'].apply(
    lambda x: 1 if x.strip().lower() == 'malignant' else 0
)

# Filtrar apenas imagens que existem
existing = [img_id for img_id in df['ID']
            if os.path.exists(os.path.join(IMAGES_DIR, img_id + '.png'))]
df = df[df['ID'].isin(existing)].reset_index(drop=True)

n_benign    = (df['label'] == 0).sum()
n_malignant = (df['label'] == 1).sum()

# Verificar máscaras
n_masks = sum(1 for img_id in df['ID']
              if os.path.exists(os.path.join(MASKS_DIR,
                                             img_id.replace('bus_', 'mask_') + '.png')))

print(f"Dataset: {len(df)} imagens ({n_benign} benignas, {n_malignant} malignas)")
print(f"Pacientes únicos (Case): {df['Case'].nunique()}")
print(f"Imagens com máscara de segmentação: {n_masks}/{len(df)}")
print(f"Configuração: {len(SEEDS)} seeds × {N_SPLITS} folds = {len(SEEDS)*N_SPLITS} runs")
print(f"\nDistribuição de classes:")
print(df['Pathology'].value_counts())'''


CELL_6_TRAINING_LOOP = r'''# ==============================================================================
# 6. LOOP PRINCIPAL: 3 Seeds × 5 Folds
# ==============================================================================

# Arquivo JSON para salvar incrementalmente
results_json_path = os.path.join(RESULTS_DIR, 'all_fold_results.json')

# Carregar resultados anteriores se existirem (para retomar após crash)
if os.path.exists(results_json_path):
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
    completed = {(r['seed'], r['fold']) for r in all_results}
    print(f"\n⚠️  Encontrados {len(all_results)} runs anteriores. Retomando de onde parou.")
else:
    all_results = []
    completed = set()

# Histórico de treino para gráficos de convergência
all_histories = []

for seed in SEEDS:
    seed_everything(seed)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], df['Case'])):

        # Pular folds já completados
        if (seed, fold) in completed:
            print(f"\nSeed {seed} | Fold {fold+1}/{N_SPLITS} — JÁ COMPLETADO, pulando.")
            continue

        print(f"\n{'='*60}")
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

        # ---- Preparar dataset do fold ----
        fold_dir = os.path.join(TEMP_DIR, f'seed{seed}_fold{fold}')
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)

        yaml_path, stats = prepare_fold_dataset(
            train_df, val_df, fold_dir, IMAGES_DIR, MASKS_DIR
        )
        print(f"  📁 Dataset: Train {stats['train']['total']} "
              f"({stats['train']['with_mask']} com máscara), "
              f"Val {stats['val']['total']} ({stats['val']['with_mask']} com máscara)")

        # ---- Treinar YOLO26n-seg (modelo fresco a cada fold!) ----
        model = YOLO('yolo26n-seg.pt')

        run_name = f'seed{seed}_fold{fold}'

        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            optimizer='AdamW',
            lr0=1e-3,
            lrf=0.01,
            weight_decay=5e-4,
            seed=seed,
            deterministic=True,
            # === DESABILITAR TODAS AS AUGMENTATIONS ===
            # (conforme decisão da pesquisadora — sem augmentation)
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            erasing=0.0,
            # Projeto e organização
            project=RESULTS_DIR,
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        # ---- Carregar melhor modelo ----
        best_model_path = os.path.join(RESULTS_DIR, run_name, 'weights', 'best.pt')
        best_model = YOLO(best_model_path)

        # ---- Métricas de CLASSIFICAÇÃO (nível de imagem) ----
        gt_lookup = dict(zip(val_df['ID'], val_df['label']))
        val_img_dir = os.path.join(fold_dir, 'images', 'val')

        cls_metrics, probs, labels = extract_classification_metrics(
            best_model, val_img_dir, gt_lookup
        )

        # ---- Métricas de SEGMENTAÇÃO (via YOLO val) ----
        val_results = best_model.val(
            data=yaml_path, imgsz=IMG_SIZE, batch=BATCH_SIZE, verbose=False
        )
        seg_metrics = {
            'seg_mAP50':    float(val_results.seg.map50),
            'seg_mAP50_95': float(val_results.seg.map),
            'box_mAP50':    float(val_results.box.map50),
            'box_mAP50_95': float(val_results.box.map),
        }

        # ---- Ler histórico de treino ----
        results_csv_path = os.path.join(RESULTS_DIR, run_name, 'results.csv')
        total_epochs = EPOCHS
        history = None
        if os.path.exists(results_csv_path):
            train_log = pd.read_csv(results_csv_path)
            train_log.columns = [c.strip() for c in train_log.columns]
            total_epochs = len(train_log)
            history = {
                'train_cls_loss': train_log.get('train/cls_loss', pd.Series()).tolist(),
                'val_cls_loss':   train_log.get('val/cls_loss', pd.Series()).tolist(),
                'train_box_loss': train_log.get('train/box_loss', pd.Series()).tolist(),
                'val_box_loss':   train_log.get('val/box_loss', pd.Series()).tolist(),
                'train_seg_loss': train_log.get('train/seg_loss', pd.Series()).tolist(),
                'val_seg_loss':   train_log.get('val/seg_loss', pd.Series()).tolist(),
            }

        # ---- Registrar resultado ----
        result = {
            'seed': seed,
            'fold': fold,
            'total_epochs': total_epochs,
            **cls_metrics,
            **seg_metrics,
            'n_val_images': len(labels),
        }
        all_results.append(result)

        # Salvar incrementalmente no JSON (no Drive!)
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Salvar histórico
        if history:
            all_histories.append({
                'seed': seed, 'fold': fold, 'history': history
            })

        print(f"  📊 Classificação — AUC: {cls_metrics['auc']:.4f} | "
              f"F1: {cls_metrics['f1']:.4f} | Sens: {cls_metrics['sensitivity']:.4f} | "
              f"Spec: {cls_metrics['specificity']:.4f}")
        print(f"  📊 Segmentação  — mAP50: {seg_metrics['seg_mAP50']:.4f} | "
              f"mAP50-95: {seg_metrics['seg_mAP50_95']:.4f}")
        if cls_metrics['n_no_detection'] > 0:
            print(f"  ⚠️  Sem detecção em {cls_metrics['n_no_detection']}/{len(labels)} imagens")
        print(f"  💾 Resultados salvos ({len(all_results)}/{len(SEEDS)*N_SPLITS} runs)")

        # Limpar diretório temporário do fold (economizar espaço)
        shutil.rmtree(fold_dir)

print(f"\n{'='*60}")
print(f"✅ TREINAMENTO COMPLETO! {len(all_results)} runs finalizados.")
print(f"{'='*60}")'''


CELL_7_RESULTS_TABLE = r'''# ==============================================================================
# 7. TABELA DE RESULTADOS
# ==============================================================================

# Carregar resultados do JSON (funciona mesmo se o kernel reiniciou)
results_json_path = os.path.join(RESULTS_DIR, 'all_fold_results.json')
with open(results_json_path, 'r') as f:
    all_results = json.load(f)

df_results = pd.DataFrame(all_results)

# Salvar como CSV
csv_path = os.path.join(RESULTS_DIR, 'resultados_YOLO26_seg.csv')
df_results.to_csv(csv_path, index=False)
print(f"CSV salvo em: {csv_path}")

# Exibir tabela de classificação
print("\n📋 MÉTRICAS DE CLASSIFICAÇÃO por Seed × Fold:")
print("=" * 110)
cls_cols = ['seed', 'fold', 'total_epochs', 'auc', 'f1', 'sensitivity',
            'specificity', 'balanced_accuracy', 'n_no_detection']
print(df_results[cls_cols].to_string(index=False, float_format='%.4f'))

# Exibir tabela de segmentação
print("\n📋 MÉTRICAS DE SEGMENTAÇÃO por Seed × Fold:")
print("=" * 80)
seg_cols = ['seed', 'fold', 'seg_mAP50', 'seg_mAP50_95', 'box_mAP50', 'box_mAP50_95']
print(df_results[seg_cols].to_string(index=False, float_format='%.4f'))

# Resumo por seed
print("\n📋 Resumo AUC por Seed:")
print("=" * 60)
for seed in SEEDS:
    seed_data = df_results[df_results['seed'] == seed]
    print(f"  Seed {seed}: AUC = {seed_data['auc'].mean():.4f} ± {seed_data['auc'].std():.4f}")

# Resumo geral
print("\n📋 RESUMO GERAL (Classificação):")
print("=" * 60)
metrics_to_report = ['auc', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'accuracy']
for m in metrics_to_report:
    vals = df_results[m]
    print(f"  {m.upper():25s}: {vals.mean():.4f} ± {vals.std():.4f}")

print("\n📋 RESUMO GERAL (Segmentação):")
print("=" * 60)
seg_report = ['seg_mAP50', 'seg_mAP50_95', 'box_mAP50', 'box_mAP50_95']
for m in seg_report:
    vals = df_results[m]
    print(f"  {m:25s}: {vals.mean():.4f} ± {vals.std():.4f}")

# Total de imagens sem detecção
total_no_det = df_results['n_no_detection'].sum()
total_val = df_results['n_val_images'].sum()
print(f"\n⚠️  Total de imagens sem detecção: {total_no_det}/{total_val} "
      f"({100*total_no_det/total_val:.1f}%)")'''


CELL_8_VISUALIZATIONS = r'''# ==============================================================================
# 8. VISUALIZAÇÕES
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('YOLO26n-seg — Resultados 5-Fold CV × 3 Seeds', fontsize=14, fontweight='bold')

# 8a. Boxplot de AUC por Seed
ax = axes[0, 0]
seed_aucs = [df_results[df_results['seed'] == s]['auc'].values for s in SEEDS]
bp = ax.boxplot(seed_aucs, labels=[str(s) for s in SEEDS], patch_artist=True)
colors = ['#3498db', '#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xlabel('Seed')
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC por Seed (Classificação)')
ax.axhline(y=df_results['auc'].mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'Média: {df_results["auc"].mean():.4f}')
ax.legend()

# 8b. Barplot de métricas de classificação
ax = axes[0, 1]
metrics_names = ['AUC', 'F1', 'Sens.', 'Spec.', 'Bal.Acc.']
metrics_keys  = ['auc', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy']
means = [df_results[k].mean() for k in metrics_keys]
stds  = [df_results[k].std() for k in metrics_keys]
bars = ax.bar(metrics_names, means, yerr=stds, capsize=5,
              color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'], alpha=0.7)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Valor')
ax.set_title('Métricas de Classificação (média ± std)')
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# 8c. Convergência (último fold treinado)
ax = axes[1, 0]
if len(all_histories) > 0:
    last_h = all_histories[-1]['history']
    epochs_range = range(1, len(last_h.get('train_cls_loss', [])) + 1)
    if len(epochs_range) > 0:
        ax.plot(epochs_range, last_h['train_cls_loss'], label='Train cls_loss', color='#3498db')
        ax.plot(epochs_range, last_h['val_cls_loss'], label='Val cls_loss', color='#e74c3c')
        ax.set_xlabel('Época')
        ax.set_ylabel('Classification Loss')
        ax.set_title(f'Convergência (Seed {all_histories[-1]["seed"]}, '
                     f'Fold {all_histories[-1]["fold"]+1})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Histórico vazio', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Convergência')
else:
    ax.text(0.5, 0.5, 'Histórico não disponível\n(kernel reiniciou?)',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Convergência')

# 8d. Barplot de métricas de segmentação
ax = axes[1, 1]
seg_names = ['Seg mAP50', 'Seg mAP50-95', 'Box mAP50', 'Box mAP50-95']
seg_keys  = ['seg_mAP50', 'seg_mAP50_95', 'box_mAP50', 'box_mAP50_95']
seg_means = [df_results[k].mean() for k in seg_keys]
seg_stds  = [df_results[k].std() for k in seg_keys]
bars = ax.bar(seg_names, seg_means, yerr=seg_stds, capsize=5,
              color=['#1abc9c', '#16a085', '#2980b9', '#2c3e50'], alpha=0.7)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Valor')
ax.set_title('Métricas de Segmentação (média ± std)')
for bar, mean in zip(bars, seg_means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'resultados_visuais.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico salvo em: {os.path.join(RESULTS_DIR, 'resultados_visuais.png')}")'''


CELL_9_BOOTSTRAP_CI = r'''# ==============================================================================
# 9. INTERVALO DE CONFIANÇA 95% (Bootstrap)
# ==============================================================================

def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Calcula IC via bootstrap percentile."""
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

print("📊 RESUMO FINAL — CLASSIFICAÇÃO COM IC 95% (Bootstrap)")
print("=" * 70)
cls_report = [
    ('AUC-ROC (primária)',  'auc'),
    ('F1',                  'f1'),
    ('Sensibilidade',       'sensitivity'),
    ('Especificidade',      'specificity'),
    ('Balanced Accuracy',   'balanced_accuracy'),
    ('Accuracy',            'accuracy'),
]

summary_rows = []
for name, key in cls_report:
    vals = df_results[key].values
    mean = vals.mean()
    std  = vals.std()
    lo, hi = bootstrap_ci(vals)
    print(f"  {name:25s}: {mean:.4f} ± {std:.4f}  [IC 95%: {lo:.4f} — {hi:.4f}]")
    summary_rows.append({
        'Métrica': name, 'Média': mean, 'Std': std,
        'IC95_lower': lo, 'IC95_upper': hi
    })

print("\n📊 RESUMO FINAL — SEGMENTAÇÃO COM IC 95% (Bootstrap)")
print("=" * 70)
seg_report = [
    ('Seg mAP50',     'seg_mAP50'),
    ('Seg mAP50-95',  'seg_mAP50_95'),
    ('Box mAP50',     'box_mAP50'),
    ('Box mAP50-95',  'box_mAP50_95'),
]

for name, key in seg_report:
    vals = df_results[key].values
    mean = vals.mean()
    std  = vals.std()
    lo, hi = bootstrap_ci(vals)
    print(f"  {name:25s}: {mean:.4f} ± {std:.4f}  [IC 95%: {lo:.4f} — {hi:.4f}]")
    summary_rows.append({
        'Métrica': name, 'Média': mean, 'Std': std,
        'IC95_lower': lo, 'IC95_upper': hi
    })

# Salvar resumo
df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(RESULTS_DIR, 'resumo_YOLO26_seg.csv')
df_summary.to_csv(summary_path, index=False)
print(f"\n💾 Resumo salvo em: {summary_path}")'''


# =============================================================================
# NOTEBOOK ASSEMBLY
# =============================================================================

def make_yolo26_seg_notebook():
    """Assemble all cells into a notebook."""
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

    nb['cells'] = [
        nbf.v4.new_markdown_cell(MARKDOWN_TITLE),
        nbf.v4.new_code_cell(CELL_1_DRIVE),
        nbf.v4.new_code_cell(CELL_2_IMPORTS),
        nbf.v4.new_code_cell(CELL_3_CONFIG),
        nbf.v4.new_code_cell(CELL_4_FUNCTIONS),
        nbf.v4.new_code_cell(CELL_5_LOAD_CSV),
        nbf.v4.new_code_cell(CELL_6_TRAINING_LOOP),
        nbf.v4.new_code_cell(CELL_7_RESULTS_TABLE),
        nbf.v4.new_code_cell(CELL_8_VISUALIZATIONS),
        nbf.v4.new_code_cell(CELL_9_BOOTSTRAP_CI),
    ]

    return nb


if __name__ == '__main__':
    nb = make_yolo26_seg_notebook()
    with open('YOLO26_Seg_Training.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("✅ YOLO26_Seg_Training.ipynb gerado com sucesso!")
    print("   → Faça upload para o Google Colab e execute com GPU habilitada.")
