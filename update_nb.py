import json

notebook_path = "shortcut_test.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update imports and add easyocr reader
        if "import cv2" in source and "import easyocr" not in source:
            new_source = source.replace("import cv2\n", "import cv2\nimport easyocr\n\n# Initialize OCR reader globally to avoid reloading the model per image\nreader = easyocr.Reader(['en'], gpu=True)\n")
            
            # Since source is a string, we split it back to lines keeping the newlines for Jupyter format
            lines = new_source.splitlines(True)
            cell['source'] = lines
            
        # 2. Update extract_artifacts_only function
        if "def extract_artifacts_only(" in source:
            new_func = """def extract_artifacts_only(image, mask, **kwargs):
    \"\"\"
    Opção B: Extrai artefatos sobrepostos (calipers, texto, medidas) usando 
    OCR para textos numéricos/alfabéticos e filtragem de formas (cruzes) para calipers.
    
    Args:
        image:            np.ndarray float64 [0,1], shape (H, W)
        mask:             np.ndarray bool — máscara da lesão
        **kwargs:         parâmetros antigos mantidos por compatibilidade com extract_combined
        
    Returns:
        np.ndarray float64 [0,1] — imagem apenas com artefatos detectados
    \"\"\"
    result = np.zeros_like(image)
    h, w = image.shape
    
    # Converter para uint8 para EasyOCR e OpenCV
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.dtype == np.float64 else np.clip(image, 0, 255).astype(np.uint8)
    
    # --- 1. OCR (Textos) ---
    detections = reader.readtext(img_uint8)
    for bbox, text, prob in detections:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w_box, h_box = cv2.boundingRect(pts)
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + w_box + pad), min(h, y + h_box + pad)
        result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        
    # --- 2. Calipers (Shapes/Cruzes) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(tophat, 150, 255, cv2.THRESH_BINARY)
    
    if mask.max() > 0:
        mask_dilated = _get_dilated_mask(mask, PERILESIONAL_DILATION)
        thresh[mask_dilated > 0] = 0
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box)/h_box if h_box > 0 else 0
        
        if 5 < area < 200 and 0.5 < aspect_ratio < 2.0:
            pad = 3
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + w_box + pad), min(h, y + h_box + pad)
            result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            
    return result

"""
            # Replace the old function block. 
            # We need to find the start of extract_artifacts_only and the start of the next function
            start_idx = source.find("def extract_artifacts_only(")
            end_idx = source.find("def extract_combined(", start_idx)
            
            if start_idx != -1 and end_idx != -1:
                new_source = source[:start_idx] + new_func + source[end_idx:]
                cell['source'] = new_source.splitlines(True)


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully.")
