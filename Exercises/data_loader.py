from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import csv


@dataclass
class SimpleDataFile:
    path: str | Path
    delimiter: str | None = None

    _headers: list[str] | None = None
    _data: np.ndarray | None = None
    _header_index: dict[str, int] | None = None

    def __post_init__(self):
        self.path = Path(self.path)

    # ---------- (b) factory method ----------
    @classmethod
    def read_file(cls, path: str | Path, delimiter: str | None = None) -> "SimpleDataFile":
        obj = cls(path=path, delimiter=delimiter)
        obj. load()
        return obj

    # ----------- (f) lazy loading ----------
    def _ensure_loaded(self):
        if self._data is None or self._headers is None or self._header_index is None:
            self._load()

    def _detect_delimiter(self, line: str) -> str:
        return "," if "," in line else None

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        text = self.path.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Empty file")

        header_line = lines[0]

        if self.delimiter is None:
            delim = self._detect_delimiter(header_line)
        else:
            delim = self.delimiter

        reader = csv.reader(lines, delimiter=delim,
                            quotechar='"', skipinitialspace=True)
        rows = list(reader)

        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]

        if headers and (headers[0] == "" or headers[0].lower().startswith("unnamed")):
            headers = headers[1:]
            data_rows = [r[1:] for r in data_rows]

        data = []
        for r in data_rows:
            if len(r) != len(headers):
                raise ValueError(
                    f"Row has {len(r)} columns but header has {len(headers)} columns.\nRow: {r}")
            clean = [x.strip().strip('"').strip("'") for x in r]
            data.append([float(x) for x in clean])

        self._headers = headers
        self._data = np.array(data, dtype=float)
        self._header_index = {name: i for i, name in enumerate(headers)}

    # ---------- (d) header index ----------
    @property
    def headers(self) -> list[str]:
        self._ensure_loaded()
        return self._headers

    def col_index(self, name: str) -> int:
        self._ensure_loaded()
        if name not in self._header_index:
            raise KeyError(
                f"Unknown column: {name}, Available: {list(self._header_index.keys())}")
        return self._header_index[name]

    # ---------- (c) row/col extraction ----------
    def row(self, i: int) -> np.ndarray:
        self._ensure_loaded()
        return self._data[i, :]

    def col(self, key: int | str) -> np.ndarray:
        self._ensure_loaded()
        j = key if isinstance(key, int) else self.col_index(key)
        return self._data[:, j]

    def shape(self) -> tuple[int, int]:
        self._ensure_loaded()
        return self._data.shape

    # ---------- (e) pretty print ----------
    def pretty(self, n: int = 5) -> str:
        self._ensure_loaded()
        n = min(n, self._data.shape[0])
        out = []
        out.append(f"File: {self.path}")
        out.append(f"Shape: {self._data.shape}")
        out.append("Columns:")
        for h in self._headers:
            vals = self.col(h)[:n]
            out.append(f" -{h}: {vals}")
        return "\n".join(out)

    def __repr__(self) -> str:
        try:
            return self.pretty(n=3)
        except Exception:
            return f"SimpleDataFile(path={self.path!s})"
