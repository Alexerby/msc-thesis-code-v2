from __future__ import annotations

"""Centralised SOEP‑loader registry.

This module hides every table‑name / column‑list detail behind a single
interface so that higher‑level code (e.g. `BafoegCalculator`) only speaks
in domain language (ppath, pl, …) and never touches I/O specifics again.
"""

from typing import Dict, List, Tuple

from data_handler import SOEPDataHandler

# ---------------------------------------------------------------------------
# Table specification
# ---------------------------------------------------------------------------
# key -> (filename in SOEP archive, list_of_columns_to_read)
_SPEC: Dict[str, Tuple[str, List[str]]] = {
    "ppath": (
        "ppathl",
        [
            "pid",
            "hid",
            "syear",
            "gebjahr",
            "sex",
            "gebmonat",
            "parid",
            "partner",
        ],
    ),
    "pl": (
        "pl",
        ["pid", "syear", "plg0012_h", "plh0258_h", "plc0168_h"],
    ),
    "pgen": (
        "pgen",
        ["pid", "syear", "pglabgro"],
    ),
    "bioparen": (
        "bioparen",
        ["pid", "fnr", "mnr"],
    ),
    "region": (
        "regionl",
        ["hid", "bula", "syear"],
    ),
    "hgen": (
        "hgen",
        ["hid", "hgtyp1hh", "syear"],
    ),
}


class LoaderRegistry:
    """Lazy‑loading façade around all SOEP sheets we need.

    Example
    -------
    >>> loaders = LoaderRegistry()
    >>> df_students = loaders.ppath()       # loads only ppath on first call
    >>> loaders.load_all()                  # force all remaining sheets in one go
    """

    # Dynamically create one SOEPDataHandler per spec entry
    def __init__(self) -> None:
        self._handlers: Dict[str, SOEPDataHandler] = {
            key: SOEPDataHandler(filename) for key, (filename, _) in _SPEC.items()
        }
        # Cache for already‑loaded DataFrames
        self.data: Dict[str, "pd.DataFrame"] = {}

    # ---------------------------------------------------------------------
    # Generic helpers
    # ---------------------------------------------------------------------
    def load(self, key: str):
        """Load *one* sheet and return a fresh DataFrame copy."""
        if key in self.data:  # already loaded → return cached copy
            return self.data[key]

        filename, cols = _SPEC[key]
        handler = self._handlers[key]
        handler.load_dataset(cols)
        # Always store a copy so caller mutations don’t corrupt registry state
        self.data[key] = handler.data.copy()
        return self.data[key]

    def load_all(self):
        """Eagerly load every sheet defined in the spec."""
        for key in _SPEC.keys():
            self.load(key)

    # ---------------------------------------------------------------------
    # Convenience getters (so callers don’t need magic strings)
    # ---------------------------------------------------------------------
    def ppath(self):
        return self.load("ppath")

    def pl(self):
        return self.load("pl")

    def pgen(self):
        return self.load("pgen")

    def bioparen(self):
        return self.load("bioparen")

    def region(self):
        return self.load("region")

    def hgen(self):
        return self.load("hgen")

    # ---------------------------------------------------------------------
    # Dunder methods for convenience / debugging
    # ---------------------------------------------------------------------
    def __getitem__(self, key: str):
        """Dictionary‑style access to already‑loaded sheets."""
        return self.data[key]

    def __contains__(self, key: str):
        return key in self.data

    def __iter__(self):
        return iter(self.data.items())
