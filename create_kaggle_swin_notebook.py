import json
import os

input_file = '/home/giovanna/Documents/Unifei/IC/Breast_analysis/breast_ultrasound_anomalies/SwinViT_Training.ipynb'
output_file = '/home/giovanna/Documents/Unifei/IC/Breast_analysis/breast_ultrasound_anomalies/SwinViT_Training_Kaggle.ipynb'

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modifying the first markdown cell (index 0)
nb['cells'][0]['source'][-1] = "> Os resultados são salvos localmente em `/kaggle/working/`. Ao final do treinamento, lembre-se de baixar o arquivo gerado pelo Kaggle."

# Modifying the first code cell (index 1)
new_source_cell_1 = [
    "# ==============================================================================\n",
    "# 1. DEFINIR CAMINHOS (KAGGLE)\n",
    "# ==============================================================================\n",
    "import os\n",
    "\n",
    "# ---- EDITE AQUI se o nome do seu dataset importado for diferente ----\n",
    "BASE_DIR = '/kaggle/input/busbra-dataset' # Substitua 'busbra-dataset' pelo slug que você der no Kaggle\n",
    "# -------------------------------------------------------------------\n",
    "\n",
    "DATASET_CSV  = os.path.join(BASE_DIR, 'BUSBRA', 'BUSBRA', 'bus_data.csv')\n",
    "IMAGES_DIR   = os.path.join(BASE_DIR, 'processed', 'full')\n",
    "RESULTS_DIR  = '/kaggle/working/results/SwinViT_full'\n",
    "\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "# Verificação rápida\n",
    "assert os.path.exists(DATASET_CSV), f\"CSV não encontrado: {DATASET_CSV}\\n(Lembre-se de importar o dataset BUSBRA no Kaggle com o path correto)\"\n",
    "assert os.path.isdir(IMAGES_DIR),   f\"Diretório de imagens não encontrado: {IMAGES_DIR}\"\n",
    "print(f\"✅ CSV: {DATASET_CSV}\")\n",
    "print(f\"✅ Imagens: {IMAGES_DIR}\")\n",
    "print(f\"✅ Resultados serão salvos em: {RESULTS_DIR}\")"
]

nb['cells'][1]['source'] = new_source_cell_1

# Check the training cell (index 6) to make sure saving text doesn't say "no Drive!"
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "(no Drive!)" in line:
                line = line.replace("(no Drive!)", "(local no Kaggle!)")
            new_source.append(line)
        cell['source'] = new_source

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Created Kaggle Notebook:", output_file)
