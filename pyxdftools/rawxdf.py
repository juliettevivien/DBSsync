"""Thin wrapper for raw XDF data."""

import pyxdf

from .errors import DataStreamLoadError


class RawXdf:
    """Thin wrapper for raw XDF data.

    Provides convenience methods for accessing the raw XDF
    list-of-dictionaries data structure.

    Attributes:
        filename: XDF file - string or Path.
    """

    filename = None

    # Boolean determining additional logging.
    __verbose = None

    # Boolean indicating if file has been loaded.
    __loaded = False

    # Dictionary containing file header information.
    __header = None

    # List of dictionaries corresponding to each loaded stream.
    __streams = None

    def __init__(self, filename, verbose=False):
        """Initialise XDF file."""
        self.filename = filename
        self.verbose = verbose

    def resolve_streams(self):
        """Resolve streams in the current file."""
        return pyxdf.resolve_streams(str(self.filename))

    def available_stream_ids(self):
        """Return a list of available stream IDs."""
        streams = RawXdf.resolve_streams(self)
        stream_ids = sorted([stream['stream_id'] for stream in streams])
        print(stream_ids)
        return stream_ids

    def load(self, **kwargs):
        """Load XDF data using pyxdf passing all kwargs."""
        try:
            streams, header = pyxdf.load_xdf(self.filename, **kwargs)
        except Exception:
            streams, header = self.__failsafe_load(**kwargs)

        # Initialise class attributes.
        self.__loaded = True
        self.__header = header
        self.__streams = streams
        return self

    def __failsafe_load(self, **kwargs):
        if 'select_streams' in kwargs:
            stream_ids = kwargs.pop('select_streams')
        else:
            stream_ids = self.available_stream_ids()

        # Test loading each stream.
        loadable_streams = []
        for i in stream_ids:
            try:
                _, _ = pyxdf.load_xdf(self.filename, select_streams=[i],
                                      **kwargs)
                loadable_streams.append(i)
            except Exception as exc:
                exc = DataStreamLoadError(i, exc)
                print(exc)
        return pyxdf.load_xdf(self.filename, select_streams=loadable_streams,
                              **kwargs)

    def loaded(self):
        """Test if a file has been loaded."""
        return self.__loaded

    def assert_loaded(self):
        """Assert that data is loaded before continuing."""
        if not self.loaded():
            raise UserWarning(
                'No streams loaded, call load_streams() first.')

    def get_header(self):
        """Return the raw header info dictionary."""
        self.assert_loaded()
        return self.__header['info']

    def num_loaded_streams(self):
        """Return the number of streams currently loaded."""
        self.assert_loaded()
        return len(self.__streams)

    def loaded_stream_ids(self):
        """Get IDs for all loaded streams."""
        return sorted([self.__get_stream_id(stream)
                       for stream in self.__streams])

    def get_streams(self, *stream_ids):
        """Return raw stream data.

        Select streams according to their ID or default all loaded
        streams.
        """
        self.assert_loaded()
        # If no stream_ids are provided return all loaded streams.
        if not stream_ids:
            return self.__streams
        self.__check_stream_ids(*stream_ids)
        return [stream for stream_id in stream_ids
                for stream in self.__streams
                if self.__get_stream_id(stream) == stream_id]

    def collect_stream_data(self, *stream_ids, data_path=None,
                            pop_singleton_lists=False,
                            as_key=False):
        """Extract nested stream data for multiple streams.

        Returns a dictionary {stream_id: data} with number of items
        equal to the number of streams. If no data is available at any
        key in the data path the item value will be None.
        """
        streams = self.get_streams(*stream_ids)
        data = {self.__get_stream_id(stream):
                self.__get_stream_data(stream, data_path,
                                       as_key=as_key)
                for stream in streams}
        if pop_singleton_lists:
            data = self.__pop_singleton_lists(data)
        return data

    def collect_leaf_data(self, data, leaf_data=None):
        """Collect singleton items of metadata in leaf nodes."""
        if leaf_data is None:
            leaf_data = {}
        for key, item in data.items():
            if isinstance(item, dict):
                self.collect_leaf_data(item, leaf_data)
            if isinstance(item, list):
                if len(item) == 1:
                    if isinstance(item[0], str) or item[0] is None:
                        leaf_data[key] = item
                    else:
                        self.collect_leaf_data(item[0], leaf_data)
        return leaf_data

    def __get_stream_data(self, stream, data_path, *, as_key=False):
        """Extract nested stream data at data_path."""
        data = stream
        for key in data_path:
            if data and key in data.keys():
                data = data[key]
                if isinstance(data, list) and len(data) == 1:
                    data = data[0]
            else:
                stream_id = self.__get_stream_id(stream)
                print(f'Stream {stream_id} does not contain key: {key}.')
                return None
        if as_key:
            data = {as_key: data}
        return data

    def __get_stream_id(self, stream):
        # Get ID for stream.
        return self.__get_stream_data(stream, ['info', 'stream_id'])

    def __check_stream_ids(self, *stream_ids):
        # Check requests streams are loaded.
        valid_ids = set(self.loaded_stream_ids()).intersection(
            stream_ids)
        try:
            assert len(valid_ids) == len(stream_ids)
        except AssertionError:
            invalid_ids = list(valid_ids.symmetric_difference(stream_ids))
            raise KeyError(f'Invalid stream IDs: {invalid_ids}')

    def __pop_singleton_lists(self, data):
        # Copy dictionary to avoid modifying in place.
        data = data.copy()
        for key, item in data.items():
            if isinstance(item, list):
                if len(item) == 1:
                    item = item[0]
                else:
                    item = [self.__pop_singleton_lists(i) for i in item]
            if isinstance(item, dict):
                item = self.__pop_singleton_lists(item)

            data[key] = item
        return data