import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import easyocr
import torch
from scipy.ndimage import binary_dilation

BASE_DIR = r"C:\Users\Bruno\Documents\Giovanna\breast_ultrasound_anomalies"
BUSBRA_DIR = os.path.join(BASE_DIR, "BUSBRA/BUSBRA")
IMAGES_DIR = os.path.join(BUSBRA_DIR, "Images")
MASKS_DIR = os.path.join(BUSBRA_DIR, "Masks")
CSV_PATH = os.path.join(BUSBRA_DIR, "bus_data.csv")

PERILESIONAL_DILATION = 30

# Keywords to detect breast side text
SIDE_KEYWORDS = ["mama", "esq", "dir", "esquerda", "direita", "left", "right", "l.", "r.", " mama "]

def _get_dilated_mask(mask, dilation_radius):
    struct = np.ones((3, 3))
    return binary_dilation(mask, structure=struct, iterations=dilation_radius)

def detect_calipers(image, mask):
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.dtype == np.float64 else np.clip(image, 0, 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    
    _, thresh = cv2.threshold(tophat, 150, 255, cv2.THRESH_BINARY)
    
    if mask.max() > 0:
        mask_dilated = _get_dilated_mask(mask, PERILESIONAL_DILATION)
        thresh[mask_dilated > 0] = 0
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caliper_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box)/h_box if h_box > 0 else 0
        
        if 5 < area < 200 and 0.5 < aspect_ratio < 2.0:
            caliper_count += 1
            
    return caliper_count > 0

def process_single_image(img_id, row_data, reader):
    img_name = f"{img_id}.png"
    mask_name = f"mask_{img_id[4:]}.png"
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    mask_path = os.path.join(MASKS_DIR, mask_name)
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None
        
    try:
        # Load image and mask
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float64) > 128
        
        # 1. Detect Calipers
        has_calipers = detect_calipers(image, mask)
        
        # 2. Run OCR
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        detections = reader.readtext(img_uint8)
        
        has_text = len(detections) > 0
        has_side_text = False
        detected_texts = []
        
        for bbox, text, prob in detections:
            txt_lower = text.lower()
            detected_texts.append(text)
            
            for kw in SIDE_KEYWORDS:
                if kw in txt_lower:
                    has_side_text = True
                    break
                    
        return {
            'ID': img_id,
            'pathology': row_data['Pathology'],
            'device': row_data['Device'],
            'has_calipers': has_calipers,
            'has_text': has_text,
            'has_side_text': has_side_text,
            'detected_texts': detected_texts
        }
    except Exception as e:
        print(f"Error processing {img_id}: {e}")
        return None

def run_analysis():
    use_gpu = torch.cuda.is_available()
    print(f"CUDA Available: {use_gpu}")
    
    # If using CPU, set threads to 4
    if not use_gpu:
        torch.set_num_threads(4)
        
    print("Loading catalog...")
    df = pd.read_csv(CSV_PATH)
    
    tasks = []
    for _, row in df.iterrows():
        img_id = row['ID']
        tasks.append((img_id, row.to_dict()))
        
    print(f"Total catalog entries to process: {len(tasks)}")
    
    print("Initializing OCR reader...")
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
    print(f"Reader initialized in {time.time() - t0:.2f} seconds.")
    
    print("Starting analysis on all images...")
    results = []
    t_start = time.time()
    for i, (img_id, row_data) in enumerate(tasks):
        res = process_single_image(img_id, row_data, reader)
        if res is not None:
            results.append(res)
        
        # Print progress every 20 images
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            ips = (i + 1) / elapsed
            eta_seconds = (len(tasks) - (i + 1)) / ips
            print(f"Processed {i + 1}/{len(tasks)} images... Speed: {ips:.2f} img/s. ETA: {eta_seconds/60:.1f} mins.")
                
    # Save raw results
    out_json = os.path.join(BASE_DIR, "processed/dataset_analysis_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Analysis complete. Processed {len(results)} images in {time.time() - t_start:.1f} seconds.")
    
    # Compute Statistics
    rdf = pd.DataFrame(results)
    
    # Generate full report
    generate_markdown_report(rdf)

def generate_markdown_report(df):
    report_path = os.path.join(BASE_DIR, "processed/dataset_analysis_report.md")
    
    total = len(df)
    benign = sum(df['pathology'] == 'benign')
    malignant = sum(df['pathology'] == 'malignant')
    
    # Calipers
    cal_benign_yes = sum((df['pathology'] == 'benign') & df['has_calipers'])
    cal_benign_no = sum((df['pathology'] == 'benign') & ~df['has_calipers'])
    cal_mal_yes = sum((df['pathology'] == 'malignant') & df['has_calipers'])
    cal_mal_no = sum((df['pathology'] == 'malignant') & ~df['has_calipers'])
    
    # Side texts
    side_benign_yes = sum((df['pathology'] == 'benign') & df['has_side_text'])
    side_benign_no = sum((df['pathology'] == 'benign') & ~df['has_side_text'])
    side_mal_yes = sum((df['pathology'] == 'malignant') & df['has_side_text'])
    side_mal_no = sum((df['pathology'] == 'malignant') & ~df['has_side_text'])

    # General texts
    text_benign_yes = sum((df['pathology'] == 'benign') & df['has_text'])
    text_benign_no = sum((df['pathology'] == 'benign') & ~df['has_text'])
    text_mal_yes = sum((df['pathology'] == 'malignant') & df['has_text'])
    text_mal_no = sum((df['pathology'] == 'malignant') & ~df['has_text'])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Análise de Viés e Atalhos (Dataset Completo - BUSBRA)\n\n")
        f.write(f"Total de imagens analisadas: **{total}**\n")
        f.write(f"- Benignas: **{benign}** ({benign/total*100:.1f}%)\n")
        f.write(f"- Malignas: **{malignant}** ({malignant/total*100:.1f}%)\n\n")
        
        f.write("## 1. Distribuição de Calipers\n")
        f.write("| Classe | Com Calipers | Sem Calipers | Total |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| Benignas | {cal_benign_yes} ({cal_benign_yes/benign*100:.1f}%) | {cal_benign_no} ({cal_benign_no/benign*100:.1f}%) | {benign} |\n")
        f.write(f"| Malignas | {cal_mal_yes} ({cal_mal_yes/malignant*100:.1f}%) | {cal_mal_no} ({cal_mal_no/malignant*100:.1f}%) | {malignant} |\n")
        f.write(f"| **Total** | **{cal_benign_yes+cal_mal_yes}** | **{cal_benign_no+cal_mal_no}** | **{total}** |\n\n")
        
        f.write("## 2. Distribuição de Textos de Lateralidade (Mama Direita/Esquerda)\n")
        f.write("Proporção de imagens que contêm termos como 'mama', 'esq', 'dir', 'esquerda', 'direita', 'left', 'right', 'l.', 'r.' gravados na imagem.\n\n")
        f.write("| Classe | Com Texto de Lateralidade | Sem Texto de Lateralidade | Total |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| Benignas | {side_benign_yes} ({side_benign_yes/benign*100:.1f}%) | {side_benign_no} ({side_benign_no/benign*100:.1f}%) | {benign} |\n")
        f.write(f"| Malignas | {side_mal_yes} ({side_mal_yes/malignant*100:.1f}%) | {side_mal_no} ({side_mal_no/malignant*100:.1f}%) | {malignant} |\n")
        f.write(f"| **Total** | **{side_benign_yes+side_mal_yes}** | **{side_benign_no+side_mal_no}** | **{total}** |\n\n")

        f.write("## 3. Distribuição de Qualquer Texto (Geral)\n")
        f.write("Presença de qualquer anotação textual (incluindo marca de aparelho, escala, profundidade e medidas do operador).\n\n")
        f.write("| Classe | Com Qualquer Texto | Sem Qualquer Texto | Total |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| Benignas | {text_benign_yes} ({text_benign_yes/benign*100:.1f}%) | {text_benign_no} ({text_benign_no/benign*100:.1f}%) | {benign} |\n")
        f.write(f"| Malignas | {text_mal_yes} ({text_mal_yes/malignant*100:.1f}%) | {text_mal_no} ({text_mal_no/malignant*100:.1f}%) | {malignant} |\n")
        f.write(f"| **Total** | **{text_benign_yes+text_mal_yes}** | **{text_benign_no+text_mal_no}** | **{total}** |\n\n")
        
        f.write("## 4. Distribuição por Equipamento (Device)\n")
        f.write("Investigação se calipers e lateralidade se concentram em marcas específicas de aparelhos de ultrassom.\n\n")
        f.write("| Equipamento | Benigno | Maligno | % Maligno | Com Calipers | Com Lateralidade |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        devices = df['device'].unique()
        for dev in devices:
            dev_df = df[df['device'] == dev]
            d_total = len(dev_df)
            if d_total == 0:
                continue
            d_ben = sum(dev_df['pathology'] == 'benign')
            d_mal = sum(dev_df['pathology'] == 'malignant')
            d_cal = sum(dev_df['has_calipers'])
            d_side = sum(dev_df['has_side_text'])
            f.write(f"| {dev} | {d_ben} | {d_mal} | {d_mal/d_total*100:.1f}% | {d_cal} ({d_cal/d_total*100:.1f}%) | {d_side} ({d_side/d_total*100:.1f}%) |\n")
        f.write("\n")
        
        f.write("## Conclusão e Insights de Viés\n")
        
        caliper_diff = abs(cal_benign_yes/benign - cal_mal_yes/malignant) * 100
        side_diff = abs(side_benign_yes/benign - side_mal_yes/malignant) * 100
        
        f.write("### Conclusões Principais:\n")
        if caliper_diff > 10:
            f.write(f"- ⚠️ **ALTO RISCO DE CALIPERS:** Há uma diferença de {caliper_diff:.1f}% na presença de calipers entre imagens benignas e malignas. A ResNet-50 pode aprender a classificar lesões com base nos calipers, agindo como um atalho indesejado.\n")
        else:
            f.write(f"- ✅ **Caliper Balanceado:** A diferença na incidência de calipers entre benignos e malignos é de apenas {caliper_diff:.1f}%, sugerindo um risco menor de atalho direto por essa característica.\n")
            
        if side_diff > 10:
            f.write(f"- ⚠️ **ALTO RISCO DE TEXTO DE LATERALIDADE:** Há uma diferença de {side_diff:.1f}% na presença de textos de lateralidade. A rede pode aprender que termos específicos da mama indicam malignidade.\n")
        else:
            f.write(f"- ✅ **Lateralidade Balanceada:** A diferença na incidência de textos de lateralidade entre as classes é de apenas {side_diff:.1f}%, minimizando o risco de viés de lateralidade.\n")
            
    print(f"Markdown report generated at {report_path}")

if __name__ == "__main__":
    run_analysis()
