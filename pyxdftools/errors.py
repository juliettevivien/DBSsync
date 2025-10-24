"""Custom pyxdf-tools errors."""


class MetadataParseError (Exception):
    """Custom exception raised when failing to parse XDF metadata."""

    def __init__(self, stream_id):
        self.stream_id = stream_id

    def __str__(self):
        return f'Stream ID: {self.stream_id}'


class DataStreamLoadError (Exception):
    """Custom exception raised when failing to load XDF stream data."""

    def __init__(self, stream_id, cause):
        self.stream_id = stream_id
        self.__cause__ = cause

    def __str__(self):
        return f'Failed to load Stream {self.stream_id}: {self.__cause__}'