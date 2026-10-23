"""
This module contains mappings from standard ORSO fields to NeXus hierarchy that are
implemented as links to the orso hdf group.
"""

from typing import List

import h5py


class NexusMapping:
    """
    Performs a linked mapping from ORSO header group to NeXus hierarchy element.
    """
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    def __call__(self, entry):
        entry[self.dest] = h5py.SoftLink(f"{entry.name}/info/{self.source}")


class SampeMapping(NexusMapping):

    def __call__(self, entry):
        # make group NXSample complient
        group = entry.create_group(self.dest)
        group.attrs["NX_class"] = "NXsample"
        group['name'] = h5py.SoftLink(f"{entry.name}/info/{self.source}/name")


NXMAPPINGS: List[NexusMapping] = [
    NexusMapping('data_source/experiment/start_date', 'start_time'),
    SampeMapping('data_source/experiment/sample', 'sample'),
]
