import os
import pandas as pd
import json


class DestatisData:
    def __init__(self, 
                 table: str, 
                 header_rows_to_skip: int = 0, 
                 header: list | int = 0,
                 footer_rows_to_skip: int = 0,
                 ) -> None:
        config_path = self._get_config_path()
        config = self._load_config(config_path)

        self.path = os.path.expanduser(config["paths"]['data']['destatis'])
        self.table_path = os.path.join(self.path, f"{table}.csv")
        self.header = [header] if isinstance(header, int) else header

        self.header_rows_to_skip = header_rows_to_skip
        self.footer_rows_to_skip = footer_rows_to_skip

        self.df = self._load_data()

    def _get_config_path(self) -> str:
        """Returns the absolute path to the config.json file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
        return config_path

    def _load_config(self, config_path: str) -> dict:
        """
        Load a JSON config
        """
        with open(config_path, 'r') as f:
            return json.load(f)


    def _load_data(self) -> pd.DataFrame:
        """Load the data from the specified CSV file, skipping rows at the top and bottom."""
        df = pd.read_csv(
            self.table_path, 
            sep=";", 
            header = self.header,
            skiprows=self.header_rows_to_skip,  
            engine="python", 
            on_bad_lines="skip"
        )

        # If footer_rows_to_skip > 0, remove last N rows
        if self.footer_rows_to_skip > 0:
            df = df.iloc[:-self.footer_rows_to_skip]  # Remove last N rows

        return df
