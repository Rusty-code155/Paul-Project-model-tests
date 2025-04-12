# Code by Turner Miles Peeples
# data_preprocessing.py
import pandas as pd
import difflib

REQUIRED_HEADERS = [
    "Ionic Fluid", "Nanoparticle", "Size (nm)", "Concentration", "Shape",
    "Temperature", "Thermal Conductivity"
]

POSSIBLE_HEADER_ALIASES = {
    "Ionic Fluid": ["Ionic Fluid", "Base Liquid", "IonicLiquid", "Solvent"],
    "Nanoparticle": ["Nanoparticle", "NP", "Particle"],
    "Size (nm)": ["Size (nm)", "Size", "Particle Size", "Diameter (nm)"],
    "Concentration": ["Concentration", "Conc", "Concentration (% wt)", "wt%", "%"],
    "Shape": ["Shape", "Orientation", "Structure", "Morphology"],
    "Temperature": ["Temperature", "Temp", "Temp (C)", "T", "Temperature (deg. K)"],
    "Thermal Conductivity": ["Thermal Conductivity", "Conductivity", "k", "λ", "k (W/mK)"]
}

def find_closest_column(columns, targets):
    if isinstance(columns, pd.Index):
        columns = columns.tolist()
    for target in targets:
        best_match = difflib.get_close_matches(target, columns, n=1, cutoff=0.5)
        if best_match:
            return best_match[0]
    return None

def clean_column_names(df):
    df.columns = [str(col).strip() for col in df.columns]
    return df

def parse_size(size_str):
    try:
        size_str = str(size_str).strip()
        print(f"Parsing Size (nm): '{size_str}'")
        if '-' in size_str:
            low, high = map(float, size_str.split('-'))
            avg = (low + high) / 2
            print(f"Parsed range {size_str} as average: {avg}")
            return avg
        parsed = float(size_str)
        print(f"Parsed single value {size_str} as: {parsed}")
        return parsed
    except (ValueError, TypeError) as e:
        print(f"Failed to parse Size (nm): '{size_str}' - Error: {e}")
        return None

def extract_data_table(df):
    matched_cols = {}
    header_row_idx = None

    # First, find the header row
    for row_idx, row in df.iterrows():
        row_values = row.fillna("").astype(str).str.strip().tolist()
        temp_matched_cols = {}
        for col_name, aliases in POSSIBLE_HEADER_ALIASES.items():
            match = find_closest_column(row_values, aliases)
            if match:
                col_index = row_values.index(match)
                temp_matched_cols[col_name] = col_index
                print(f"Matched {col_name} to column '{match}' at index {col_index}")

        if len(temp_matched_cols) >= 7:  # Found enough columns to consider this the header row
            matched_cols = temp_matched_cols
            header_row_idx = row_idx
            break

    # If we didn't find the header row, log the failures
    if not matched_cols:
        for col_name, aliases in POSSIBLE_HEADER_ALIASES.items():
            print(f"Could not find column for {col_name}. Aliases tried: {aliases}")
        raise ValueError("No matching table structure found.")

    # Extract data starting from the row after the header
    data = []
    for r in range(header_row_idx + 1, len(df)):
        row_data = df.iloc[r].tolist()
        try:
            ionic = str(row_data[matched_cols["Ionic Fluid"]]).strip()
            particle = str(row_data[matched_cols["Nanoparticle"]]).strip()
            size = parse_size(row_data[matched_cols["Size (nm)"]])
            if size is None:
                continue
            conc = float(row_data[matched_cols["Concentration"]])
            
            # Preserve the '-' in the Shape column
            shape = str(row_data[matched_cols["Shape"]]).strip()
            # TODO: If more data becomes available, revisit the Shape column to replace '-' with actual shapes
            # (e.g., confirm the shape of 50 nm Aluminum Oxide nanoparticles in [C2mim][CH3SO3]).
            print(f"Extracted Shape for row {r}: '{shape}'")
            
            temp = float(row_data[matched_cols["Temperature"]])
            cond = float(row_data[matched_cols["Thermal Conductivity"]])
            
            # Detailed logging for all columns
            print(f"Row {r}: Ionic Fluid='{ionic}', Nanoparticle='{particle}', Size={size}, Shape='{shape}', "
                  f"Concentration={conc}, Temperature={temp}, Thermal Conductivity={cond}")
        except (ValueError, TypeError):
            continue

        data.append({
            "Ionic Fluid": ionic,
            "Nanoparticle": particle,
            "Size (nm)": size,
            "Concentration": conc,
            "Shape": shape,
            "Temperature": temp,
            "Thermal Conductivity": cond
        })

    if not data:
        raise ValueError("No valid data rows found after the header.")
    return pd.DataFrame(data)

def group_dataframes(dfs):
    grouped = {}
    for df in dfs:
        try:
            table = extract_data_table(df)
            for _, group in table.groupby(["Ionic Fluid", "Nanoparticle", "Size (nm)", "Concentration", "Shape"]):
                key = tuple(group[['Ionic Fluid', 'Nanoparticle', 'Size (nm)', 'Concentration', 'Shape']].iloc[0])
                print(f"Grouped data with key: {key}")
                grouped[key] = group.reset_index(drop=True)
        except Exception as e:
            print(f"Skipping sheet due to error: {e}")
    return grouped