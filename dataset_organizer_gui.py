import os
import shutil
import glob
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Configurações de diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "BUSBRA", "BUSBRA", "Images")
DEST_DIR = os.path.join(BASE_DIR, "organizadas")

# Mapeamento de atalhos para os caminhos de destino
SHORTCUTS = {
    'c': "caliper",
    't': "texto",
    'n': "limpa",
    '1': "equipamentos/Toshiba",
    '2': "equipamentos/GE_Logiq_5",
    '3': "equipamentos/GE_Logiq_7",
    '4': "equipamentos/U-Systems",
}

class DatasetOrganizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BUSBRA Dataset Organizer")
        self.geometry("1000x800")
        
        # Carrega a lista de imagens
        self.image_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.*")))
        # Filtra por extensões comuns de imagem se necessário
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not self.image_paths:
            messagebox.showerror("Erro", f"Nenhuma imagem encontrada em: {SOURCE_DIR}")
            self.destroy()
            return
            
        self.current_index = 0
        self.current_tags = set()
        
        # Cria os diretórios de destino
        self.create_directories()
        
        # Setup UI
        self.setup_ui()
        
        # Bindings
        self.bind("<Right>", self.next_image)
        self.bind("<Left>", self.prev_image)
        for key in SHORTCUTS.keys():
            self.bind(f"<Key-{key}>", self.handle_shortcut)
        self.bind("<z>", self.undo)
        
        # Foco principal e carrega a primeira imagem
        self.focus_set()
        self.load_image()

    def create_directories(self):
        for folder in SHORTCUTS.values():
            os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

    def setup_ui(self):
        # Frame superior para informações
        self.info_frame = tk.Frame(self, bg="#2c3e50")
        self.info_frame.pack(fill=tk.X, side=tk.TOP)
        
        self.progress_label = tk.Label(self.info_frame, text="", fg="white", bg="#2c3e50", font=("Arial", 14, "bold"))
        self.progress_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.filename_label = tk.Label(self.info_frame, text="", fg="#ecf0f1", bg="#2c3e50", font=("Arial", 12))
        self.filename_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Frame central para a imagem
        self.image_label = tk.Label(self)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Frame inferior para feedback visual
        self.feedback_frame = tk.Frame(self, bg="#ecf0f1")
        self.feedback_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.tags_label = tk.Label(self.feedback_frame, text="Tags aplicadas: Nenhuma", fg="#27ae60", bg="#ecf0f1", font=("Arial", 12, "bold"))
        self.tags_label.pack(pady=5)
        
        # Instruções
        instructions = "Atalhos: C (Caliper), T (Texto), N (Limpa) | Equipamentos: 1 (Toshiba), 2 (GE 5), 3 (GE 7), 4 (U-Systems) | Z (Desfazer) | Setas (Navegar)"
        self.inst_label = tk.Label(self.feedback_frame, text=instructions, fg="#7f8c8d", bg="#ecf0f1", font=("Arial", 10))
        self.inst_label.pack(pady=2)

    def load_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            img_path = self.image_paths[self.current_index]
            filename = os.path.basename(img_path)
            
            # Atualiza informações
            self.progress_label.config(text=f"Imagem {self.current_index + 1} de {len(self.image_paths)}")
            self.filename_label.config(text=filename)
            
            # Reseta as tags aplicadas para a nova imagem (para exibição)
            self.current_tags = set()
            self.update_tags_label()
            
            # Carrega e redimensiona a imagem usando PIL
            try:
                pil_img = Image.open(img_path)
                # Pega dimensões da janela ou usa tamanho padrão
                window_width = self.winfo_width()
                window_height = self.winfo_height() - 100 # Subtrai espaço dos frames
                if window_width <= 1 or window_height <= 1:
                    window_width, window_height = 800, 600
                
                # Preserva o aspect ratio
                pil_img.thumbnail((window_width, window_height), Image.Resampling.LANCZOS)
                
                self.tk_image = ImageTk.PhotoImage(pil_img)
                self.image_label.config(image=self.tk_image)
            except Exception as e:
                self.image_label.config(image='', text=f"Erro ao carregar imagem:\n{str(e)}")
        
    def next_image(self, event=None):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
            
    def handle_shortcut(self, event):
        key = event.keysym.lower()
        if key in SHORTCUTS:
            folder = SHORTCUTS[key]
            self.copy_current_image(folder)
            
    def copy_current_image(self, dest_folder_name):
        if not (0 <= self.current_index < len(self.image_paths)):
            return
            
        src_path = self.image_paths[self.current_index]
        filename = os.path.basename(src_path)
        dest_folder_path = os.path.join(DEST_DIR, dest_folder_name)
        dest_path = os.path.join(dest_folder_path, filename)
        
        try:
            shutil.copy2(src_path, dest_path)
            # Adiciona ao set de tags da imagem atual (nome legível)
            tag_name = dest_folder_name.replace("equipamentos/", "").capitalize()
            self.current_tags.add((dest_folder_name, tag_name))
            self.update_tags_label()
        except Exception as e:
            messagebox.showerror("Erro de cópia", f"Falha ao copiar para {dest_folder_name}:\n{str(e)}")

    def undo(self, event=None):
        if not self.current_tags:
            return
            
        if not (0 <= self.current_index < len(self.image_paths)):
            return
            
        src_path = self.image_paths[self.current_index]
        filename = os.path.basename(src_path)
        
        erros = []
        for dest_folder_name, _ in self.current_tags:
            dest_path = os.path.join(DEST_DIR, dest_folder_name, filename)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except Exception as e:
                erros.append(str(e))
                
        if erros:
            messagebox.showwarning("Erro no Undo", f"Algumas cópias não puderam ser apagadas:\n{', '.join(erros)}")
        else:
            self.current_tags.clear()
            self.update_tags_label()

    def update_tags_label(self):
        if not self.current_tags:
            self.tags_label.config(text="Tags aplicadas: Nenhuma", fg="#7f8c8d")
        else:
            tags_str = ", ".join(sorted([name for _, name in self.current_tags]))
            self.tags_label.config(text=f"Tags aplicadas: {tags_str}", fg="#27ae60")
            
    # Lida com redimensionamento da janela
    def bind_resize(self):
        self.bind("<Configure>", self.on_resize)
        
    def on_resize(self, event):
        # Apenas se for redimensionamento da janela principal e não de widgets internos
        if event.widget == self:
            # debounce do resize poderia ser implementado, mas para simplificar vamos deixar apenas o on_resize normal
            # se ficar lento, podemos colocar um after()
            pass

if __name__ == "__main__":
    app = DatasetOrganizerApp()
    app.mainloop()
