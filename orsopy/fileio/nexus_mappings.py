"""
This module contains mappings from standard ORSO fields to NeXus hierarchy that are
implemented as links to the orso hdf group.
"""

from typing import List

try:
    from typing import Protocol
except ImportError:
    Protocol = object

import h5py

USE_SOFTLINKS = False


class NexusMapping(Protocol):
    def __call__(self, entry: h5py.Group): ...


class NexusValue:
    """
    Performs a linked mapping from ORSO header group to NeXus hierarchy element.
    """

    def __init__(self, dest: str, source: str):
        self.source = source
        self.dest = dest

    def __call__(self, entry: h5py.Group):
        path = f"{entry.name}/info/{self.source}"
        if path in entry:
            if USE_SOFTLINKS:
                entry[self.dest] = h5py.SoftLink(path)
            else:
                entry[self.dest] = entry[path]


class NexusGroup:
    def __init__(self, dest: str, nx_class: str, **sources):
        self.dest = dest
        self.nx_class = nx_class
        self.sources = sources

    def __call__(self, entry: h5py.Group):
        # make group NXSample complient
        group = entry.create_group(self.dest)
        group.attrs["NX_class"] = self.nx_class
        for source_name, source in self.sources.items():
            if type(source) is str:
                path = f"{entry.name}/info/{source}"
                if path in entry:
                    if USE_SOFTLINKS:
                        group[source_name] = h5py.SoftLink(path)
                    else:
                        group[source_name] = entry[path]
            else:
                # allow multiple possible locations, first one tried first
                for si in source:
                    path = f"{entry.name}/info/{si}"
                    if path in entry:
                        if USE_SOFTLINKS:
                            group[source_name] = h5py.SoftLink(path)
                        else:
                            group[source_name] = entry[path]
                        break


NXMAPPINGS: List[NexusMapping] = [
    NexusValue("title", "data_source/experiment/title"),
    NexusValue("experiment_identifier", "data_source/experiment/proposalID"),
    NexusValue("start_time", "data_source/experiment/start_date"),
    NexusGroup(
        "sample",
        "NXsample",
        name="data_source/sample/name",
        temperature="data_source/sample/temperature",
        electric_field="data_source/sample/electric_field",
        magnetic_field="data_source/sample/magnetic_field",
        description="data_source/sample/description",
    ),
    NexusGroup(
        "user",
        "NXuser",
        name="data_source/owner/name",
        affiliation="data_source/owner/affiliation",
        email="data_source/owner/contact",
    ),
    NexusGroup(
        "instrument",
        "NXinstrument",
        name="data_source/experiment/instrument",
    ),
    NexusGroup(
        "instrument/beam",
        "NXbeam",
        incident_wavelength=(
            "data_source/measurement/instrument_settings/wavelength/magnitude",
            "data_source/measurement/instrument_settings/wavelength/individual_magnitudes",
            "data_source/measurement/instrument_settings/wavelength/_mean",
        ),
    ),
    NexusGroup(
        "reduction",
        "NXprocess",
        program=("reduction/software/name"),
        version=("reduction/software/version"),
        date=("reduction/timestamp"),
    ),
]
