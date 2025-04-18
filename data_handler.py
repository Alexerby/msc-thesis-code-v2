from typing import List, Union, Dict, cast
from pathlib import Path

import pandas as pd



from misc.utility_functions import get_config_path, load_config


class SOEPDataHandler:
    """
    Class represents an individual dataset/file from SOEP-Core. 
    For instance pass in dataset "pl". Will refer to pl.csv
    """
    def __init__(self, file: Union[str, Path]):
        """
        Initialize with a file where SOEP files are stored.

        :file: The dataset/file to be represented by the class.
        """
        # Load config file
        config_path: Path = get_config_path(Path("config.json"))
        config: Dict = load_config(config_path)

        # Get the data directory from the config file
        self.data_dir = Path(config["paths"]["data"]["soep"]).expanduser().resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"SOEP data directory not found: {self.data_dir}")

        # Store the file path given during initialization
        self.file = Path(file).with_suffix(".csv")
        self.file_path = self.data_dir / self.file

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        # Placeholder for the loaded data
        self.data: pd.DataFrame = pd.DataFrame()

    @property
    def dataset_name(self) -> str:
        return self.file.stem

    def load_file(self) -> None:
        """
        Load a dataset file from the SOEP data directory.
        """
        file_path = self.data_dir / self.file
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.data = pd.read_csv(file_path)


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
