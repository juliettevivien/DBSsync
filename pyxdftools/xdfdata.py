"""Main XdfData class for working with XDF data."""


import mne
import pandas as pd
import pyxdf

from .constants import data_channel_types, microvolts
from .errors import MetadataParseError
from .rawxdf import RawXdf


class XdfData (RawXdf):
    """Helper class for with XDF data files.

    Provides a pandas-based layer of abstraction over raw XDF data to
    simplify data processing.
    """

    # DataFrame containing metadata for all loaded streams.
    __metadata = None

    __metadata_types = {
        'channel_count': int,
        'nominal_srate': float,
        'v4data_port': int,
        'v4service_port': int,
        'v6data_port': int,
        'v6service_port': int,
        'effective_srate': float,
        'first_timestamp': float,
        'last_timestamp': float,
        'sample_count': int,
    }

    def __init__(self, filename, verbose=False):
        """Initialise raw XDF."""
        super().__init__(filename, verbose)

    def resolve_streams(self, **match_props):
        """Return a DataFrame containing available streams.

        Streams can be optionally filtered using key-value matching
        properties. Available streams are selected based on matching
        property values and/or lists of property values (e.g. type='eeg'
        or stream_id=[1, 2]). Property values are matched against the
        raw XDF metadata - i.e. before metadata has been loaded and
        optionally pre-processed.
        """
        streams = pd.DataFrame(super().resolve_streams())

        # Subset streams based on matching properties.
        if len(match_props) > 0:
            # Check matching properties are valid stream properties.
            for prop in list(match_props):
                if prop not in streams.columns:
                    print(f'Property {prop} not a stream property')
                    del match_props[prop]

            # Ensure all property values are lists so we can match on
            # multiple criteria.
            match_props = {k: v if isinstance(v, list) else [v]
                           for k, v in match_props.items()}

            # Warn on non-matching property values.
            for prop, values in match_props.items():
                for value in values:
                    if not (streams[prop] == value).any():
                        print(f'Ignoring: {prop}={value}.')

            # All streams matching any property value will be
            # included.
            mask = streams.isin(match_props).any(axis='columns')
            streams = streams.loc[mask]

        streams.set_index('stream_id', inplace=True)
        streams.sort_index(inplace=True)
        return streams

    def load(self, *stream_ids, **kwargs):
        """Load XDF data from file using pyxdf.load_xdf().

        Any pyxdf.load_xdf() kwargs provided will be passed to that
        function. All other kwargs are assumed to be stream properties
        and will be passed to resolve_streams().
        """
        # Separate kwargs.
        xdf_kwargs = {k: kwargs[k] for k in
                      kwargs.keys() & pyxdf.load_xdf.__kwdefaults__.keys()}
        match_props = {k: kwargs[k] for k in
                       kwargs.keys() - pyxdf.load_xdf.__kwdefaults__.keys()}
        # Add stream_ids as matching stream properties.
        if len(stream_ids) > 0:
            match_props['stream_id'] = list(stream_ids)
        # Load matching streams from the XDF file.
        stream_ids = list(self.resolve_streams(**match_props).index)
        super().load(select_streams=stream_ids, **xdf_kwargs)

        # Parse stream metadata.
        metadata = self.parse_metadata()

        # Initialise class attributes.
        self.__metadata = metadata
        if self.verbose:
            print(f'Loaded streams: {" ".join(str(i) for i in stream_ids)}')
        return self

    def raw_xdf(self, *stream_ids):
        """Return raw XDF data for stream_ids or all loaded streams."""
        return self.get_streams(*stream_ids)

    def header(self):
        """Return XDF header as a DataFrame."""
        header = pd.DataFrame(self.get_header())
        header['datetime'] = pd.to_datetime(header['datetime'])
        return header

    def parse_metadata(self):
        """Return a DataFrame for all loaded streams.

        Called automatically when XDF data is loaded. This method can be
        implemented by a subclass for any custom parsing requirements.
        """
        streams = self.get_streams()
        metadata = [self.__parse_stream_metadata(stream) for stream in streams]
        metadata = pd.DataFrame(metadata)
        metadata.sort_index(inplace=True)
        metadata.index.name = 'stream_id'
        return metadata

    def metadata(self, *stream_ids):
        """Return stream metadata as a DataFrame.

        Get data for stream_ids or default all loaded streams.
        """
        self.assert_loaded()
        if stream_ids:
            return self.__metadata.loc[list(stream_ids)]
        else:
            return self.__metadata

    def channel_metadata(self, *stream_ids, force_id_idx=False):
        """Return a DataFrame containing channel metadata.

        Get data for stream_ids or default all loaded streams. Multiple
        streams always returns a hierarchical (multiindex) DataFrame.
        """
        # Extract channel metadata for data streams.
        ch_metadata = self.collect_stream_data(
            *stream_ids,
            data_path=['info', 'desc', 'channels', 'channel'],
            pop_singleton_lists=True)
        # Check that data streams have valid channel metadata.
        ch_metadata, empty = self.__remove_empty_streams(ch_metadata)
        if empty:
            print(f"""No channel metadata for streams: {' '.join(str(i)
            for i in sorted(list(empty.keys())))}""")
        if not ch_metadata:
            print('No channel metadata found!')
            return None
        df = self.__merge_stream_data(ch_metadata,
                                      'channel',
                                      force_id_idx=force_id_idx)
        return df

    def channel_metadata_subset(self, *stream_ids, types,
                                force_id_idx=False):
        """Return DataFrame subset of channel metadata.

        Types is a string or list of string to select returned metadata
        types.
        """
        ch_metadata = self.channel_metadata(*stream_ids,
                                            force_id_idx=force_id_idx)
        if ch_metadata is not None:
            if not isinstance(types, list):
                types = [types]
                if ch_metadata.columns.nlevels > 1:
                    types = list(set(types).intersection(
                        ch_metadata.columns.get_level_values(1)))
                    if types:
                        return ch_metadata.loc[:, (slice(None), types)]
                    else:
                        return None
                else:
                    types = list(set(types).intersection(
                        ch_metadata.columns))
                    if types:
                        return ch_metadata.loc[:, types]
                    else:
                        return None

    def channel_scaling(self, *stream_ids):
        """Return a DataFrame of channel scaling values."""
        units = self.channel_metadata_subset(*stream_ids,
                                             types='unit',
                                             force_id_idx=True)
        if units is not None:
            scaling = units.apply(
                lambda units: [1e-6 if u in microvolts else 1
                               for u in units])
            scaling.rename(columns={'unit': 'scale'}, inplace=True)
            return scaling

    def time_series(self, *stream_ids, scale_data=True,
                    set_channel_names=True):
        """Return a DataFrame containing stream time-series data.

        Get data for stream_ids or default all loaded streams.

        If set_channel_names=True then channels names will be set
        according to channel metadata labels.

        If scale_data=True then channel data will be scaled according to
        channel metadata unit. Currently this only applies to micro
        volts, which MNE expects to be volts.
        """
        data = self.collect_stream_data(*stream_ids,
                                        data_path=['time_series'])

        if scale_data:
            scalings = self.channel_scaling(*stream_ids)
            if scalings is not None:
                data = self.__scale_data(data, scalings)

        col_names = None
        if set_channel_names:
            col_names = self.channel_metadata_subset(*stream_ids,
                                                     types='label',
                                                     force_id_idx=True)
        ts = self.__merge_stream_data(data,
                                      'sample',
                                      col_index_name='channel',
                                      col_names=col_names)
        return ts

    def time_stamps(self, *stream_ids):
        """Return a DataFrame containing stream time-stamps.

        Get data for stream_ids or default all loaded streams.
        """
        ts = self.collect_stream_data(*stream_ids,
                                      data_path=['time_stamps'],
                                      as_key='time_stamp')
        ts = self.__merge_stream_data(ts, 'sample')
        return ts

    def data(self, *stream_ids, scale_data=True,
             set_channel_names=True, time_stamps=True):
        """Return a DataFrame containing stream data.

        Get data for stream_ids or default all loaded streams. The
        time_stamps=False this is just an alias for time_series().
        """
        ts = self.time_series(*stream_ids,
                              scale_data=scale_data,
                              set_channel_names=set_channel_names)
        if time_stamps:
            times = self.time_stamps(*stream_ids)
            ts = ts.join(times)
            if ts.columns.nlevels > 1:
                ts.sort_index(axis='columns', level='stream_id',
                              sort_remaining=False, inplace=True)
        return ts

    def clock_offsets(self, *stream_ids):
        """Return a DataFrame containing clock offset data.

        Get offset data for stream_ids or default to all loaded streams.
        """
        clock_offsets = self.collect_stream_data(
            *stream_ids,
            data_path=['footer', 'info', 'clock_offsets', 'offset'],
            pop_singleton_lists=True)
        df = self.__merge_stream_data(clock_offsets, 'sample')
        return df

    def raw_mne(self, *stream_ids):
        """Return mne.io.Raw objects from XDF streams.

        For a single stream return an mne.io.Raw object, otherwise a
        list of mne.io.Raw objects.
        """
        if len(stream_ids) == 0:
            # Return all loaded streams.
            stream_ids = self.loaded_stream_ids()
        if len(stream_ids) == 1:
            return self.__xdf_to_mne(*stream_ids)
        else:
            return [self.__xdf_to_mne(stream_id) for stream_id in stream_ids]

    def __xdf_to_mne(self, stream_id):
        fs = self.metadata(stream_id).nominal_srate.iloc[0]
        channels = self.channel_metadata(stream_id)
        ts = self.time_series(stream_id).T
        info = mne.create_info(list(channels.label),
                               fs,
                               list(channels.type))
        raw = mne.io.RawArray(ts, info)
        return raw


    # Private methods.
    def __parse_stream_metadata(self, stream):
        # The XDF metadata scheme is extensible so this can not handle
        # every possibility. It tries to get all standard info
        # properties and extract what it finds in leaf nodes within
        # 'desc' and 'footer' metadata.
        try:
            info = stream['info'].copy()
            desc = info.pop('desc')[0]
            metadata = pd.DataFrame(info)
            if desc is not None:
                # Remove channel metadata to be handled separately.
                if 'channels' in desc:
                    desc = desc.copy()
                    desc.pop('channels')
                desc = pd.DataFrame(self.collect_leaf_data(desc))
                metadata = metadata.join(desc)
            footer = stream['footer']['info']
            footer = pd.DataFrame(self.collect_leaf_data(footer))
            metadata = metadata.join(footer)
            metadata = metadata.astype(self.__metadata_types)
            metadata.set_index('stream_id', drop=False, inplace=True)
            return metadata.iloc[0]
        except Exception:
            raise MetadataParseError(info['stream_id'])

    def __scale_data(self, data, scalings):
        data = {stream_id: d * scalings[stream_id].to_numpy().T
                if stream_id in scalings.columns else d
                for stream_id, d in data.items()}
        return data

    def __merge_stream_data(self, data, index_name, *, col_index_name=None,
                            col_names=None, force_id_idx=False):
        # For single streams return a non-hierarchical DataFrame unless
        # force_id_idx=True.
        if len(data) == 1 and not force_id_idx:
            stream_id = list(data.keys())[0]
            data = list(data.values())[0]
            data = self.__to_df(stream_id, data, index_name, col_index_name,
                                col_names)
            return data

        # Otherwise returns a hierarchical (MultiIndex) DataFrame with a
        # two-dimensional column index. Stream ID is the first dimension
        # of the multi-level column index.
        data = {stream_id: self.__to_df(stream_id, d,
                                        index_name,
                                        col_index_name,
                                        col_names)
                for stream_id, d, in data.items()}
        data = pd.concat(data, axis='columns')
        # Set stream_id as the first column index level.
        data.columns.set_names('stream_id', level=0, inplace=True)
        data.sort_index(axis='columns', level='stream_id',
                        sort_remaining=False, inplace=True)
        return data

    def __remove_empty_streams(self, data):
        streams = {}
        empty = {}
        for stream_id, d in data.items():
            if d is not None:
                streams[stream_id] = d
            else:
                empty[stream_id] = d
        return streams, empty

    def __to_df(self, stream_id, data, index_name, col_index_name,
                col_names=None):
        # Column names must be a multi-indexed column DataFrame where
        # level 0=stream_id and level 1=labels. Each row maps a
        # channel index to a label.
        data = pd.DataFrame(data)
        if data.empty:
            return None
        data.index.set_names(index_name, inplace=True)
        if col_index_name:
            data.columns.set_names(col_index_name, inplace=True)
        if col_names is not None and stream_id in col_names.columns:
            data.rename(columns=col_names.loc[:, stream_id]['label'],
                        inplace=True)
        return data