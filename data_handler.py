from typing import List, Union, Dict, Tuple, cast
from pathlib import Path

import pandas as pd

from misc.utility_functions import get_config_path, load_config


def resolve_dataset_path(
    file: Union[str, Path],
    config_section: str
) -> Tuple[Path, Path]:
    """
    Resolves the full file path for a dataset given its name and config section.
    """
    config_path = get_config_path(Path("config.json"))
    config: Dict = load_config(config_path)

    data_dir = Path(config["paths"]["data"][config_section]).expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file = Path(file).with_suffix(".csv")
    file_path = data_dir / file

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return data_dir, file_path



class DatasetLoader:
    def __init__(self, file: Union[str, Path], config_section: str) -> None:
        self.data_dir, self.file_path = resolve_dataset_path(file, config_section)
        self.file = Path(file).with_suffix(".csv")
        self.data: pd.DataFrame = pd.DataFrame()

    @property
    def dataset_name(self) -> str:
        return self.file.stem

    def load_dataset(self, columns, chunk_size: int = 10000, filetype: str = "csv"):
        """
        Load a dataset file in chunks from the SOEP data directory with a simple progress indicator.
        """
        file_path = self.data_dir / self.file
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        if filetype == "csv":
            # Use chunksize to read large CSV files in manageable chunks
            chunks = pd.read_csv(file_path, chunksize=chunk_size, usecols=columns)
            chunk_count = 0

            for chunk_count, chunk in enumerate(chunks, start=1):
                print(f"Processing chunk {chunk_count}...", end="\r", flush=True)
                self.data = pd.concat([self.data, chunk], ignore_index=True)
            print("File loading complete.")
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

    def load_first_n_rows(self, columns: List[str], n: int = 10000):
        """Load the first n rows of a dataset file."""
        self.data = pd.read_csv(file_path, nrows=n, usecols=columns) #type: ignore

    def filter_data(self, variables: Union[str, List[str]], filter_values: List):
        """
        Filters out rows where the specified variable(s) contain values in `filter_values`.
        
        Parameters:
            variables (str or list of str): Column name(s) to filter.
            filter_values (list): Values to exclude from the dataframe.
        """
        self.data = cast(pd.DataFrame, self.data)  

        if isinstance(variables, str):
            variables = [variables]

        for var in variables:
            if var not in self.data.columns:
                raise ValueError(f"Column '{var}' not found in data.")
            before = len(self.data)
            self.data = self.data.loc[~self.data[var].isin(filter_values)].copy()
            after = len(self.data)
            print(f"Filtered '{var}': {before - after} rows removed, {after} remaining.")


class SOEPStatutoryInputs(DatasetLoader):
    def __init__(self, file: Union[str, Path]):
        super().__init__(file, config_section="self_composed")


class SOEPDataHandler(DatasetLoader):
    """
    Class represents an individual dataset/file from SOEP-Core. 
    For instance pass in dataset "pl". Will refer to pl.csv
    """

    def __init__(self, file: Union[str, Path]):
        super().__init__(file, config_section="soep")

    def _apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mappings to the dataframe based on the existing SOEP-core mapping file (CSV).
        This will replace column values with the corresponding labels from the CSV mappings file.
        """
        # Construct the mapping file path
        mapping_file = self.data_dir / f"{self.dataset_name}_values.csv"
        
        # Read the mappings file into a DataFrame
        df_mappings = pd.read_csv(mapping_file)

        # Group mappings by variable for fast access
        mappings_by_variable = {
            var: dict(zip(group["value"].astype(str), group["label_de"]))
            for var, group in df_mappings.groupby("variable")
        }

        for column in df.columns:
            if column in mappings_by_variable and column not in ["pid", "cid", "hid", "syear"]:
                df[column] = df[column].astype(str)  # Ensure consistent dtype

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')
        
        return df

