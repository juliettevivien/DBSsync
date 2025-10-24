from .xdfdata import XdfData


class AntXdfData (XdfData):
    """Helper class for working with AntNeuro XDF data files.

    Provides a pandas-based layer of abstraction over raw XDF data to
    simplify data processing.
    """

    def parse_metadata(self):
        """Rename AntNeuro stream types when parsing metadata."""
        df = super(AntXdfData, self).parse_metadata()
        df = self.rename_stream_types(df, 'type')
        return df

    def rename_stream_types(self, df, selection):
        """Rename stream types."""
        df.loc[:,
               selection
               ] = df.loc[:,
                          selection
                          ].replace({
                              'EEG': 'eeg',
                              })
        return df

    def channel_metadata(self, *stream_ids, force_id_idx=False):
        """Return a DataFrame containing channel metadata.

        Get data for stream_ids or default all loaded streams. Multiple
        streams always returns a hierarchical (multiindex) DataFrame.
        """
        df = super(AntXdfData, self).channel_metadata(
            *stream_ids,
            force_id_idx=force_id_idx)

        if df is None:
            return

        # Remap channel names.
        df.rename(columns={'index': 'label'}, inplace=True)

        if df.columns.nlevels == 1:
            df = self.rename_channels(df, 'label')
            df = self.rename_channel_types(df, 'type')
        else:
            df = self.rename_channels(df, (slice(None), 'label'))
            df = self.rename_channel_types(df, (slice(None), 'type'))

        return df

    def rename_channels(self, df, selection):
        """Rename channels."""
        df.loc[:,
               selection
               ] = df.loc[:,
                          selection
                          ].replace({
                              '0': 'Fp1',
                              '1': 'Fpz',
                              '2': 'Fp2',
                              '3': 'F7',
                              '4': 'F3',
                              '5': 'Fz',
                              '6': 'F4',
                              '7': 'F8',
                              '8': 'FC5',
                              '9': 'FC1',
                              '10': 'FC2',
                              '11': 'FC6',
                              '12': 'M1',
                              '13': 'T7',
                              '14': 'C3',
                              '15': 'Cz',
                              '16': 'C4',
                              '17': 'T8',
                              '18': 'M2',
                              '19': 'CP5',
                              '20': 'CP1',
                              '21': 'CP2',
                              '22': 'CP6',
                              '23': 'P7',
                              '24': 'P3',
                              '25': 'Pz',
                              '26': 'P4',
                              '27': 'P8',
                              '28': 'POz',
                              '29': 'O1',
                              '30': 'Oz',
                              '31': 'O2',
                              '32': 'CPz',  # EEG 101 with 34 channels?
                              '67': 'CPz',
                              '33': 'trigger',
                              '34': 'counter',
                          })
        return df

    def rename_channel_types(self, df, selection):
        """Rename channel types."""
        df.loc[:,
               selection
               ] = df.loc[:,
                          selection
                          ].replace({
                              'ref': 'eeg',
                              'aux': 'misc',
                              'bip': 'misc',
                              'trigger': 'misc',
                              'counter': 'misc'})
        return df