import os 
import json
from typing import Dict, Literal
from pathlib import Path
import pandas as pd




def get_config_path(filename: Path) -> Path:
    """
    Returns the absolute path to the config file,
    assuming it is located in a 'config' folder one level above this file.
    """
    this_dir = Path(os.path.abspath(__file__)).parent
    parent_dir = this_dir.parent
    path = parent_dir / "config" / filename
    return path

def load_config(config_path: Path) -> Dict:
    """
    Load a JSON config.

    :param config_path: Path to the JSON configuration file
    :return: Dictionary of the config
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config

def export_data(
    data_type: Literal["csv", "excel"],
    df: pd.DataFrame,
    output_filename: str,
    sheet_name: str = "Sheet1"
):
    """Exports data to the results directory without overwriting existing files.
    
    Args:
        data_type (str): "csv" or "excel"
        df (pd.DataFrame): The DataFrame to export
        output_filename (str): Base name for the output file (no extension needed)
        sheet_name (str): Optional sheet name for Excel exports
    """

    if df is None:
        raise ValueError("Dataframe is empty")

    config_path = get_config_path(Path("config.json"))
    config = load_config(config_path)
    results_folder = Path(config["paths"]["results"]["dataframes"])

    # Ensure output directory exists
    results_folder = results_folder.expanduser().resolve()
    results_folder.mkdir(parents=True, exist_ok=True)

    # Append correct extension
    ext = ".csv" if data_type == "csv" else ".xlsx"
    output_filename = f"{output_filename}{ext}" if not output_filename.endswith(ext) else output_filename

    # Avoid overwrite
    file_path = results_folder / output_filename
    file_base, file_extension = os.path.splitext(output_filename)
    counter = 1
    while file_path.exists():
        file_path = results_folder / f"{file_base} ({counter}){file_extension}"
        counter += 1

    # Export
    if data_type == "csv":
        df.to_csv(file_path, index=False)
    elif data_type == "excel":
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
