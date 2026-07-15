import json

notebook_path = "shortcut_test.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Fix the function signature
        if "def extract_artifacts_only(image, mask, **kwargs):" in source:
            new_source = source.replace(
                "def extract_artifacts_only(image, mask, **kwargs):",
                "def extract_artifacts_only(image, mask, *args, **kwargs):"
            )
            # Since source is a string, we split it back to lines keeping the newlines for Jupyter format
            lines = new_source.splitlines(True)
            cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook fixed successfully.")
