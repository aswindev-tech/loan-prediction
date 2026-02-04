import json

nb_path = "work.ipynb"

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print("Code cells from work.ipynb:\n")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            print(f"--- Cell {i} ---")
            print(source[:500]) # Print first 500 chars to avoid huge output
            print("\n")
except Exception as e:
    print(e)
