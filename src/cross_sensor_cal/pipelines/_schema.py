"""Helpers for canonicalising Arrow tables prior to Parquet export."""

from __future__ import annotations

import pyarrow as pa

__all__ = ["to_canonical_table"]


def to_canonical_table(table: pa.Table) -> pa.Table:
    """Return a Parquet-ready Arrow table, validating minimal requirements."""

    if not isinstance(table, pa.Table):
        raise TypeError("Expected a pyarrow.Table instance")
    return table

