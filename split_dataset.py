import os
import shutil
import random
import glob

# Define uma semente para que a divisão seja aleatória mas reproduzível
random.seed(42)

base_dir = "BUSI_division"

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
masks_dir = os.path.join(base_dir, "masks")

splits = ['train', 'val', 'test']

# Cria as subpastas train, val e test dentro de images, labels e masks
for folder in [images_dir, labels_dir, masks_dir]:
    for split in splits:
        os.makedirs(os.path.join(folder, split), exist_ok=True)

# Pega todas as imagens originais
all_files_in_images = os.listdir(images_dir)
images = [f for f in all_files_in_images if f.endswith('.png')]

# Embaralha as imagens
random.shuffle(images)

# Calcula as quantidades (70% train, 15% val, 15% test)
total = len(images)
train_end = int(0.70 * total)
val_end = int(0.85 * total)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

split_mapping = {
    'train': train_imgs,
    'val': val_imgs,
    'test': test_imgs
}

for split, img_list in split_mapping.items():
    for img_name in img_list:
        base_name = img_name.replace('.png', '')
        
        # 1. Mover a imagem
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(images_dir, split, img_name)
        shutil.move(src_img, dst_img)
        
        # 2. Encontrar as máscaras e labels associadas
        # Lembre-se: benign (1).png -> a máscara é benign (1)_mask.png ou benign (1)_mask_1.png
        mask_prefix = base_name + "_mask"
        
        # Move as máscaras
        masks_found = glob.glob(os.path.join(masks_dir, mask_prefix + "*.png"))
        for m in masks_found:
            m_name = os.path.basename(m)
            shutil.move(m, os.path.join(masks_dir, split, m_name))
            
        # 3. Consolidação dos Labels para o YOLO
        # O YOLO exige que o label da imagem seja exatamente `base_name.txt`
        # e contenha todos os polígonos dentro dele.
        txts_found = glob.glob(os.path.join(labels_dir, mask_prefix + "*.txt"))
        
        # Cria o txt unificado dentro da pasta do split (ex: labels/train/benign (1).txt)
        combined_txt_path = os.path.join(labels_dir, split, base_name + ".txt")
        
        with open(combined_txt_path, 'w') as out_f:
            for t in txts_found:
                with open(t, 'r') as in_f:
                    out_f.write(in_f.read())
                # Deleta o txt fragmentado, pois não será mais usado
                os.remove(t)
                
        # Se for uma imagem da classe 'normal' e não tiver label, criamos um txt vazio.
        if not txts_found:
            open(combined_txt_path, 'w').close()

print("Dataset separado com sucesso!")
print(f"Total de imagens: {total}")
print(f"Train: {len(train_imgs)} imagens ({(len(train_imgs)/total)*100:.1f}%)")
print(f"Val: {len(val_imgs)} imagens ({(len(val_imgs)/total)*100:.1f}%)")
print(f"Test: {len(test_imgs)} imagens ({(len(test_imgs)/total)*100:.1f}%)")
