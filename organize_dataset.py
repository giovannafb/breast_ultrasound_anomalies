import os
import shutil

src_dir = 'Dataset_BUSI_with_GT'
dst_dir = 'BUSI_division'
images_dir = os.path.join(dst_dir, 'images')
masks_dir = os.path.join(dst_dir, 'masks')

# Cria as pastas de destino
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Percorre todo o diretório original
count_images = 0
count_masks = 0

for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            src_path = os.path.join(root, file)
            
            # Se tiver "_mask" no nome, vai pra pasta masks/, senão vai pra images/
            if '_mask' in file:
                dst_path = os.path.join(masks_dir, file)
                count_masks += 1
            else:
                dst_path = os.path.join(images_dir, file)
                count_images += 1
                
            shutil.copy(src_path, dst_path)

print(f"Organização concluída!")
print(f"Imagens originais movidas: {count_images}")
print(f"Máscaras movidas: {count_masks}")
