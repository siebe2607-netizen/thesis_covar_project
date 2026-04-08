import json

with open('ThesisMSc_FINAL.ipynb', 'r') as f:
    nb = json.load(f)

old_str_p = """    q=0.05, 
    horizon=1, 
    scale_features=True, """

new_str_p = """    q=0.05, 
    horizon=1, 
    window=100,
    scale_features=True, """

old_str_s = """    quantiles=[0.01, 0.05, 0.10], 
    horizons=[1, 5, 10], 
    scale_features=True, """

new_str_s = """    quantiles=[0.01, 0.05, 0.10], 
    horizons=[1, 5, 10], 
    window=100,
    scale_features=True, """

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        updated = False
        if old_str_p in source:
            source = source.replace(old_str_p, new_str_p)
            updated = True
        if old_str_s in source:
            source = source.replace(old_str_s, new_str_s)
            updated = True
            
        if updated:
            # Format the JSON structure back cleanly
            lines = []
            for line in source.splitlines(keepends=True):
                lines.append(line)
            cell['source'] = lines

with open('ThesisMSc_FINAL.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with explicit window parameter in the cells.")
