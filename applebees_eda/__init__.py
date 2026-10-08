"""Local, reproducible discount exploration; outputs are not fraud labels."""

from .pipeline import Config, ingest, open_database, prepare_facts, quality_report

__all__ = ["Config", "ingest", "open_database", "prepare_facts", "quality_report"]
