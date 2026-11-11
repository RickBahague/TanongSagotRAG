"""Exceptions shared across modules (CLI, server, pipeline)."""


class NotIndexed(Exception):
    """A collection has no chunks yet — run ingest first."""
