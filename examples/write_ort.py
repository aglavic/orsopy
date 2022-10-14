"""
This is an example of how to generate a valid ORT text format file using the orsopy library.
In most situations the parameters will be programmatically filled and not hard coded.
"""

from datetime import datetime

import numpy as np
import yaml

from orsopy import fileio
from orsopy.fileio.base import _validate_header_data


def main():
    owner = fileio.Person(name="Peter Mustermann", affiliation="Great Institute", contact="write.to.me@mail.com")
    experiment = fileio.Experiment(
        title="Greatest Beam on Earth",
        instrument="That Reflectometer",
        start_date=datetime.now(),
        probe="x-ray",
        facility="GBI",
        proposalID="123456",
    )
    sample = fileio.Sample(
        name="S1a",
        type="solid/liquid",
        composition="Si/Fe(100)/Si(50)/D2O",
        comment="Nothing to say here, but nice I could do it in principle.",
    )
    inst_set = fileio.InstrumentSettings(
        incident_angle=fileio.ValueRange(min=0.0, max=2.5, unit="deg"),
        wavelength=fileio.Value(1.54, unit="Angstrom"),
        polarization="unpolarized",
    )
    measurement = fileio.Measurement(
        instrument_settings=inst_set, data_files=["f1_001.file", "f2_003.file"], scheme="angle-dispersive"
    )
    columns = [
        fileio.Column("Qz", unit="1/angstrom"),
        fileio.Column("R", dimension="Normalized reflectivity with footprint correctd"),
        fileio.ErrorColumn("R"),
    ]

    orso_header = fileio.Orso(
        data_source=fileio.DataSource(owner, experiment, sample, measurement),
        reduction=fileio.Reduction(software="combine text files"),
        columns=columns,
    )

    # orsopy is not failing if the schema is not met exactly, validate the result:
    _validate_header_data([yaml.safe_load(orso_header.to_yaml())])

    q = np.linspace(0, 0.356, 500)
    R = np.minimum((q - 0.1) ** -4, 1.0)
    dR = np.sqrt(R) / 1e3
    R += dR * np.random.randn(R.shape[0])
    data = np.vstack([q, R, dR]).T
    fileio.save_orso([fileio.OrsoDataset(orso_header, data)], "write.ort")


if __name__ == "__main__":
    main()
