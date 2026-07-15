import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10.0'},
}
cells = []

# =========================================================================
# CELL 0: Title
# =========================================================================
cells.append(nbf.v4.new_markdown_cell("""# Análise Estatística — Protocolo do Plano Ultra-Top-Master 3.0

Este notebook implementa o protocolo estatístico completo:

| Teste | Descrição |
|-------|-----------|
| **Wilcoxon pareado** | Comparação primária (AUC-ROC) entre ResNet-50 e Swin-T nos mesmos folds |
| **Holm-Bonferroni** | Correção para comparações múltiplas nas métricas secundárias |

> **Pré-requisito:** Os dois notebooks de treinamento (`ResNet50_Training.ipynb` e `SwinViT_Training.ipynb`) devem ter sido executados e seus CSVs salvos no Google Drive."""))

# =========================================================================
# CELL 1: Setup
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 1. SETUP
# ==============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

BASE_DIR = '/content/drive/MyDrive/breast_ultrasound_anomalies'
RESNET_CSV = os.path.join(BASE_DIR, 'results', 'ResNet50_full', 'resultados_ResNet50.csv')
SWIN_CSV   = os.path.join(BASE_DIR, 'results', 'SwinViT_full', 'resultados_SwinViT.csv')

assert os.path.exists(RESNET_CSV), f"Resultados ResNet não encontrados: {RESNET_CSV}"
assert os.path.exists(SWIN_CSV),   f"Resultados SwinViT não encontrados: {SWIN_CSV}"
print("✅ CSVs encontrados!")"""))

# =========================================================================
# CELL 2: Load & Merge
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 2. CARREGAR E PAREAR RESULTADOS
# ==============================================================================
df_resnet = pd.read_csv(RESNET_CSV)
df_swin   = pd.read_csv(SWIN_CSV)

print(f"ResNet-50: {len(df_resnet)} runs")
print(f"Swin-T:    {len(df_swin)} runs")

# Merge pareado: cada linha = mesma (seed, fold)
df = pd.merge(df_resnet, df_swin, on=['seed', 'fold'], suffixes=('_resnet', '_swin'))
print(f"\\nRuns pareados: {len(df)}")
assert len(df) == 15, f"Esperados 15 runs pareados (3 seeds × 5 folds), encontrados {len(df)}"

# Exibir tabela de comparação de AUC
print("\\n📋 AUC-ROC pareada por (Seed, Fold):")
print("-" * 50)
for _, row in df.iterrows():
    diff = row['auc_resnet'] - row['auc_swin']
    winner = '← ResNet' if diff > 0 else '→ Swin' if diff < 0 else '= Empate'
    print(f"  Seed {int(row['seed']):4d} Fold {int(row['fold'])+1} | "
          f"ResNet {row['auc_resnet']:.4f} vs Swin {row['auc_swin']:.4f} | "
          f"Δ={diff:+.4f} {winner}")"""))

# =========================================================================
# CELL 3: Wilcoxon Primary
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 3. COMPARAÇÃO PRIMÁRIA: AUC-ROC (Wilcoxon Pareado)
# ==============================================================================
# O plano define: "Comparação primária: teste pareado nos mesmos folds (Wilcoxon pareado)"

auc_resnet = df['auc_resnet'].values
auc_swin   = df['auc_swin'].values

# Wilcoxon signed-rank test (bilateral)
stat, p_value_auc = wilcoxon(auc_resnet, auc_swin, alternative='two-sided')

print("=" * 60)
print("COMPARAÇÃO PRIMÁRIA: AUC-ROC")
print("=" * 60)
print(f"  Teste:      Wilcoxon signed-rank (pareado, bilateral)")
print(f"  N pareados: {len(auc_resnet)}")
print(f"  ResNet-50:  {auc_resnet.mean():.4f} ± {auc_resnet.std():.4f}")
print(f"  Swin-T:     {auc_swin.mean():.4f} ± {auc_swin.std():.4f}")
print(f"  Estatística W: {stat:.4f}")
print(f"  P-valor:    {p_value_auc:.6f}")
print()

alpha = 0.05
if p_value_auc < alpha:
    winner = "ResNet-50" if auc_resnet.mean() > auc_swin.mean() else "Swin-T"
    print(f"  ✅ RESULTADO: Diferença ESTATISTICAMENTE SIGNIFICATIVA (p < {alpha})")
    print(f"     Vantagem para: {winner}")
else:
    print(f"  ⚪ RESULTADO: NÃO há diferença estatisticamente significativa (p ≥ {alpha})")
    print(f"     Os dois modelos têm desempenho equivalente em AUC-ROC.")"""))

# =========================================================================
# CELL 4: Holm-Bonferroni Secondary
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 4. COMPARAÇÕES SECUNDÁRIAS: Holm-Bonferroni
# ==============================================================================
# O plano define: "Comparações secundárias: correção de Holm–Bonferroni"
# Métricas secundárias: sensibilidade, especificidade, F1, balanced accuracy

secondary_metrics = {
    'Sensibilidade':     ('sensitivity_resnet',       'sensitivity_swin'),
    'Especificidade':    ('specificity_resnet',       'specificity_swin'),
    'F1':                ('f1_resnet',                'f1_swin'),
    'Balanced Accuracy': ('balanced_accuracy_resnet', 'balanced_accuracy_swin'),
}

print("=" * 70)
print("COMPARAÇÕES SECUNDÁRIAS (com correção de Holm-Bonferroni)")
print("=" * 70)

# Passo 1: Wilcoxon pareado para cada métrica secundária
p_values_raw = []
metric_names = []
metric_details = []

for name, (col_r, col_s) in secondary_metrics.items():
    vals_r = df[col_r].values
    vals_s = df[col_s].values

    try:
        stat, p = wilcoxon(vals_r, vals_s, alternative='two-sided')
    except ValueError:
        # Todos iguais — sem diferença
        stat, p = 0.0, 1.0

    p_values_raw.append(p)
    metric_names.append(name)
    metric_details.append({
        'name': name,
        'resnet_mean': vals_r.mean(),
        'resnet_std':  vals_r.std(),
        'swin_mean':   vals_s.mean(),
        'swin_std':    vals_s.std(),
        'p_raw': p,
        'stat': stat,
    })

# Passo 2: Correção de Holm-Bonferroni
reject, p_corrected, _, _ = multipletests(p_values_raw, alpha=0.05, method='holm')

# Exibir resultados
for i, detail in enumerate(metric_details):
    sig = "✅ SIG." if reject[i] else "⚪ n.s."
    print(f"\\n  {detail['name'].upper()}")
    print(f"    ResNet-50: {detail['resnet_mean']:.4f} ± {detail['resnet_std']:.4f}")
    print(f"    Swin-T:    {detail['swin_mean']:.4f} ± {detail['swin_std']:.4f}")
    print(f"    p bruto:     {detail['p_raw']:.6f}")
    print(f"    p corrigido: {p_corrected[i]:.6f}")
    print(f"    {sig}")

print("\\n" + "-" * 70)
print("Nota: 'SIG.' = rejeita H₀ a α=0.05 após correção Holm-Bonferroni")
print("      'n.s.' = não significativo")"""))

# =========================================================================
# CELL 5: Summary Table
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 5. TABELA RESUMO FINAL
# ==============================================================================

print("\\n" + "=" * 90)
print("TABELA RESUMO PARA O ARTIGO")
print("=" * 90)
print(f"{'Métrica':25s} | {'ResNet-50':20s} | {'Swin-T':20s} | {'p-valor':12s} | {'Sig.':5s}")
print("-" * 90)

# AUC (primária — sem correção)
print(f"{'AUC-ROC (primária)':25s} | "
      f"{auc_resnet.mean():.4f} ± {auc_resnet.std():.4f}     | "
      f"{auc_swin.mean():.4f} ± {auc_swin.std():.4f}     | "
      f"{p_value_auc:.6f}   | "
      f"{'✅' if p_value_auc < 0.05 else '⚪':5s}")

# Secundárias (com Holm-Bonferroni)
for i, detail in enumerate(metric_details):
    print(f"{detail['name']:25s} | "
          f"{detail['resnet_mean']:.4f} ± {detail['resnet_std']:.4f}     | "
          f"{detail['swin_mean']:.4f} ± {detail['swin_std']:.4f}     | "
          f"{p_corrected[i]:.6f}   | "
          f"{'✅' if reject[i] else '⚪':5s}")

print("-" * 90)
print("Nota: p-valores secundários com correção Holm-Bonferroni.")
print("      ✅ = significativo (p < 0.05)  |  ⚪ = não significativo")

# Salvar tabela
results = []
results.append({
    'Métrica': 'AUC-ROC (primária)',
    'ResNet50_mean': auc_resnet.mean(), 'ResNet50_std': auc_resnet.std(),
    'SwinT_mean': auc_swin.mean(), 'SwinT_std': auc_swin.std(),
    'p_valor': p_value_auc, 'p_corrigido': p_value_auc, 'significativo': p_value_auc < 0.05
})
for i, detail in enumerate(metric_details):
    results.append({
        'Métrica': detail['name'],
        'ResNet50_mean': detail['resnet_mean'], 'ResNet50_std': detail['resnet_std'],
        'SwinT_mean': detail['swin_mean'], 'SwinT_std': detail['swin_std'],
        'p_valor': detail['p_raw'], 'p_corrigido': p_corrected[i], 'significativo': reject[i]
    })

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
df_stats = pd.DataFrame(results)
stats_path = os.path.join(RESULTS_DIR, 'analise_estatistica_ResNet50_vs_SwinViT.csv')
df_stats.to_csv(stats_path, index=False)
print(f"\\n💾 Tabela salva em: {stats_path}")"""))

# =========================================================================
# CELL 6: Visualization
# =========================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 6. VISUALIZAÇÃO COMPARATIVA
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ResNet-50 vs Swin-T — Comparação Estatística', fontsize=14, fontweight='bold')

# 6a. AUC side-by-side boxplot
ax = axes[0]
data = [auc_resnet, auc_swin]
bp = ax.boxplot(data, labels=['ResNet-50', 'Swin-T'], patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax.set_ylabel('AUC-ROC')
ax.set_title(f'AUC-ROC (Wilcoxon p={p_value_auc:.4f})')
# Conectar pares com linhas
for r, s in zip(auc_resnet, auc_swin):
    ax.plot([1, 2], [r, s], 'o-', color='gray', alpha=0.3, markersize=4)

# 6b. Diferenças pareadas
ax = axes[1]
diffs = auc_resnet - auc_swin
colors = ['#3498db' if d > 0 else '#e74c3c' for d in diffs]
ax.barh(range(len(diffs)), diffs, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Δ AUC (ResNet - Swin)')
ax.set_ylabel('Run (Seed × Fold)')
ax.set_title('Diferenças pareadas')
ax.set_yticks(range(len(diffs)))
ax.set_yticklabels([f"S{int(r['seed'])} F{int(r['fold'])+1}" for _, r in df.iterrows()], fontsize=8)

plt.tight_layout()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
plt.savefig(os.path.join(RESULTS_DIR, 'comparacao_estatistica.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico salvo!")"""))

nb['cells'] = cells

with open('Statistical_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("✅ Statistical_Analysis.ipynb gerado!")
