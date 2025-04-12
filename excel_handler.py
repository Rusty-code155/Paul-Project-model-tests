#Code By Turner Miles Peeples
# excel_handler.py
import os
import pandas as pd

def load_excel_files_from_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_excel(filepath)
            df.original_filename = filename
            files.append(df)
    return files

def parse_size(size_str):
    try:
        if '-' in str(size_str):
            low, high = map(float, size_str.split('-'))
            return (low + high) / 2
        return float(size_str)
    except (ValueError, TypeError):
        return None

def get_group_id(df):
    required_cols = {'Ionic Fluid', 'Nanoparticle', 'Size (nm)', 'Concentration', 'Shape'}
    if required_cols.issubset(df.columns):
        sample = df.iloc[0]
        size = parse_size(sample['Size (nm)'])
        if size is None:
            return ("Unknown",)
        return (
            str(sample['Ionic Fluid']),
            str(sample['Nanoparticle']),
            size,
            float(sample['Concentration']),
            str(sample['Shape'])
        )
    return ("Unknown",)

def save_predictions_to_excel(df, predictions, group_id, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    df['Predicted Thermal Conductivity'] = predictions
    if hasattr(df, 'original_filename'):
        base_name, ext = os.path.splitext(df.original_filename)
        filename = f"{output_folder}/{base_name}_prediction{ext}"
    else:
        safe_group_id = "_".join(str(g) for g in group_id)
        safe_group_id = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in safe_group_id)
        filename = f"{output_folder}/predictions_{safe_group_id}.xlsx"
    df.to_excel(filename, index=False)
    return filename