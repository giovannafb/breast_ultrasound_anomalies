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
from multiprocessing import Pool, cpu_count

BASE_DIR = "/home/giovanna/Documents/Unifei/IC/Breast_analysis/breast_ultrasound_anomalies"
BUSBRA_DIR = os.path.join(BASE_DIR, "BUSBRA/BUSBRA")
IMAGES_DIR = os.path.join(BUSBRA_DIR, "Images")
MASKS_DIR = os.path.join(BUSBRA_DIR, "Masks")
CSV_PATH = os.path.join(BUSBRA_DIR, "bus_data.csv")

def run_benchmark():
    # Set thread limit
    torch.set_num_threads(1)
    
    df = pd.read_csv(CSV_PATH)
    test_ids = df['ID'].head(50).tolist()
    
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("Reader init time:", time.time() - t0)
    
    t1 = time.time()
    count = 0
    for img_id in test_ids:
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            res = reader.readtext(img)
            count += 1
            
    total_time = time.time() - t1
    print(f"Processed {count} images in {total_time:.2f} seconds.")
    print(f"Average time per image (sequential, 1 thread): {total_time/count:.2f} seconds.")

if __name__ == "__main__":
    run_benchmark()
