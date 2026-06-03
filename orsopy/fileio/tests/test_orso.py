"""
Tests for fileio module
"""

import unittest

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from orsopy import fileio as fileio
from orsopy.fileio.base import (Column, File, Header, ORSOSchemaWarning, Person, Value, ValueRange, _read_header_data,
                                _validate_header_data)
from orsopy.fileio.data_source import DataSource, Experiment, InstrumentSettings, Measurement, Polarization, Sample
from orsopy.fileio.orso import Orso, OrsoDataset
from orsopy.fileio.reduction import Reduction, Software

pth = Path(__file__).absolute().parent


class TestOrso(unittest.TestCase):
    """
    Testing the Orso class.
    """

    def test_creation(self):
        """
        Creation of Orso object.
        """
        e = Experiment("Experiment 1", "ESTIA", datetime(2021, 7, 7, 16, 31, 10), "neutron")
        s = Sample("The sample")
        inst = InstrumentSettings(Value(4.0, "deg"), ValueRange(2.0, 12.0, "angstrom"))
        df = [File("README.rst", None)]
        m = Measurement(inst, df, scheme="angle-dispersive")
        p = Person("A Person", "Some Uni")
        ds = DataSource(p, e, s, m)

        soft = Software("orsopy", "0.0.1", "macOS-10.15")
        p2 = Person("Andrew McCluskey", "European Spallation Source")
        redn = Reduction(soft, datetime(2021, 7, 14, 10, 10, 10), p2, ["footprint", "background"])

        cols = [Column("Qz", unit="1/angstrom"), Column("R")]
        value = Orso(ds, redn, cols, 0)

        ds = value.data_source
        dsm = ds.measurement
        assert ds.owner.name == "A Person"
        assert dsm.data_files[0].file == "README.rst"
        assert dsm.instrument_settings.incident_angle.magnitude == 4.0
        assert dsm.instrument_settings.wavelength.min == 2.0
        assert dsm.instrument_settings.wavelength.max == 12.0
        assert value.reduction.software.name == "orsopy"
        assert value.reduction.software.version == "0.0.1"
        assert value.reduction.timestamp == datetime(2021, 7, 14, 10, 10, 10)
        assert value.columns[0].name == "Qz"
        assert value.columns[1].name == "R"
        assert value.data_set == 0

        h = value.to_yaml()
        h = "\n".join(
            ["# ORSO reflectivity data file | 0.1 standard | YAML encoding" " | https://www.reflectometry.org/", h]
        )
        g = yaml.safe_load_all(h)
        _validate_header_data([next(g)])

    def test_creation_data_set1(self):
        """
        Creation of Orso object with a non-zero data_set.
        """
        e = Experiment("Experiment 1", "ESTIA", datetime(2021, 7, 7, 16, 31, 10), "neutron")
        s = Sample("The sample")
        inst = InstrumentSettings(Value(4.0, "deg"), ValueRange(2.0, 12.0, "angstrom"))
        df = [File("README.rst", None)]
        m = Measurement(inst, df, scheme="angle-dispersive")
        p = Person("A Person", "Some Uni")
        ds = DataSource(p, e, s, m)

        soft = Software("orsopy", "0.0.1", "macOS-10.15")
        p2 = Person("Andrew McCluskey", "European Spallation Source")
        redn = Reduction(soft, datetime(2021, 7, 14, 10, 10, 10), p2, ["footprint", "background"])

        cols = [Column("Qz", unit="1/angstrom"), Column("R")]
        value = Orso(ds, redn, cols, 1)

        dsm = value.data_source.measurement
        assert value.data_source.owner.name == "A Person"
        assert dsm.data_files[0].file == "README.rst"
        assert value.reduction.software.name == "orsopy"
        assert value.columns[0].name == "Qz"
        assert value.data_set == 1

        # check that data_set can also be a string.
        value = Orso(ds, redn, cols, "fokdoks")
        assert value.data_set == "fokdoks"
        # don't want class construction coercing a str to an int
        value = Orso(ds, redn, cols, "1")
        assert value.data_set == "1"

    def test_repr(self):
        ds = fileio.Orso.empty()
        repr(ds)

    def test_write_read(self):
        # test write and read of multiple datasets
        info = fileio.Orso.empty()
        info2 = fileio.Orso.empty()
        data = np.zeros((100, 3))
        data[:] = np.arange(100.0)[:, None]

        info.columns = [
            fileio.Column("Qz", "1/angstrom"),
            fileio.Column("R"),
            fileio.ErrorColumn("R"),
        ]
        info2.columns = info.columns
        info.data_source.measurement.instrument_settings.polarization = fileio.Polarization.po
        info2.data_source.measurement.instrument_settings.polarization = fileio.Polarization.mo
        info.data_set = "up polarization"
        info2.data_set = "down polarization"
        info.data_source.comment = "test comment"
        info2.data_source.comment = "test comment"
        info2.data_source.sample.comment = "this is a comment"

        ds = fileio.OrsoDataset(info, data)
        ds2 = fileio.OrsoDataset(info2, data)

        info3 = fileio.Orso(
            data_source=fileio.DataSource(
                sample=fileio.Sample(name="My Sample", category="solid", description="Something descriptive",),
                experiment=fileio.Experiment(
                    title="Main experiment", instrument="Reflectometer", start_date=datetime.now(), probe="x-ray",
                ),
                owner=fileio.Person("someone", "important"),
                measurement=fileio.Measurement(
                    instrument_settings=fileio.InstrumentSettings(
                        incident_angle=fileio.Value(13.4, "deg"), wavelength=fileio.Value(5.34, "A"),
                    ),
                    data_files=["abc", "def", "ghi"],
                    scheme="angle-dispersive",
                ),
            ),
            reduction=fileio.Reduction(software=fileio.Software("awesome orso")),
            data_set="Filled header",
            columns=info.columns,
        )
        info3.data_source.measurement.references = ["more", "files"]
        info3.data_source.comment = None
        ds3 = fileio.OrsoDataset(info3, data)

        # .ort read/write
        fileio.save_orso([ds, ds2, ds3], "test.ort", comment="Interdiffusion")

        ls1, ls2, ls3 = fileio.load_orso("test.ort")
        assert ls1 == ds
        assert ls2 == ds2
        assert ls3 == ds3

        # .orb read/write
        fileio.save_nexus([ds, ds2, ds3], "test.orb", comment="Interdiffusion")

        ls1, ls2, ls3 = fileio.load_nexus("test.orb")
        assert ls1 == ds
        assert ls2 == ds2
        assert ls3 == ds3

        # test empty lines between datasets
        fileio.save_orso([ds, ds2, ds3], "test.ort", data_separator="\n\n")

        ls1, ls2, ls3 = fileio.load_orso("test.ort")
        assert ls1 == ds
        assert ls2 == ds2
        assert ls3 == ds3

        _read_header_data("test.ort", validate=True)

        with pytest.raises(ValueError):
            # test wrong data_separator characters
            fileio.save_orso([ds, ds2, ds3], "test.ort", data_separator="\na\n")

    def test_unique_dataset(self):
        # checks that data_set is unique on saving of OrsoDatasets
        info = Orso.empty()
        info.data_set = 0
        info.columns = [Column("stuff")] * 4

        info2 = Orso.empty()
        info2.data_set = 0
        info2.columns = [Column("stuff")] * 4

        ds = OrsoDataset(info, np.empty((2, 4)))
        ds2 = OrsoDataset(info2, np.empty((2, 4)))

        with pytest.raises(ValueError):
            fileio.save_orso([ds, ds2], "test_data_set.ort")

        with pytest.raises(ValueError):
            OrsoDataset(info, np.empty((2, 5)))

    def test_user_data(self):
        # test write and read of userdata
        info = fileio.Orso.empty()
        info.columns = [
            fileio.Column("Qz", "1/angstrom"),
            fileio.Column("R"),
            fileio.ErrorColumn("R"),
        ]

        data = np.zeros((100, 3))
        data[:] = np.arange(100.0)[:, None]
        info.ci = 1
        info.foo = ["bar", 1, 2, 3.4]
        ds = fileio.OrsoDataset(info, data)

        fileio.save_orso([ds], "test2.ort")
        ls = fileio.load_orso("test2.ort")
        assert ls[0].info.user_data == info.user_data

        # create from dictionary
        info2 = fileio.Orso(**info.to_dict())
        assert info2 == info

        # user data in sub-key
        info.data_source.test_entry = "test"
        fileio.save_orso([ds], "test2.ort")
        ls = fileio.load_orso("test2.ort")
        assert ls[0].info.user_data == info.user_data

        # create with keyword argument
        assert not hasattr(info2.data_source, "test_entry")
        ds_dict = info.data_source.to_dict()
        assert "test_entry" in ds_dict
        info2.data_source = fileio.DataSource.from_dict(ds_dict)
        assert hasattr(info2.data_source, "test_entry")

    def test_extra_elements(self):
        # if there are extra elements present in the ORT file they should still
        # be loadable. They won't be there as dataclass fields, but they'll be
        # visible as attributes.
        datasets = fileio.load_orso(pth / "test_example.ort")
        info = datasets[0].info
        assert hasattr(info.data_source.measurement.instrument_settings.incident_angle, "resolution")

    def test_save_numpy_scalar_dtypes(self):
        info = fileio.Orso.empty()
        info.data_source.measurement.instrument_settings.wavelength = Value(np.float64(10.0))
        info.data_source.measurement.instrument_settings.incident_angle = Value(np.int32(2))
        ds = fileio.orso.OrsoDataset(info, np.arange(20.0).reshape(10, 2))
        # .ort test:
        fileio.save_orso([ds], "test_numpy.ort")
        ls = fileio.load_orso("test_numpy.ort")
        i_s = ls[0].info.data_source.measurement.instrument_settings
        assert i_s.wavelength.magnitude == 10.0
        assert i_s.incident_angle.magnitude == 2
        # .orb test:
        fileio.save_nexus([ds], "test_numpy.orb")
        ln = fileio.load_nexus("test_numpy.orb")
        i_n = ln[0].info.data_source.measurement.instrument_settings
        assert i_n.wavelength.magnitude == 10.0
        assert i_n.incident_angle.magnitude == 2

    def test_nxs_special_cases(self):
        @dataclass(repr=False)
        class TestNxs(Header):
            test: list
            test2: tuple
            test3: dict

        res = TestNxs([None, {"a": "b"}], ({"a": "b"},), {})
        bio = BytesIO()
        with h5py.File(bio, "w") as f:
            res.to_nexus(f, name="test")
        res = TestNxs([None, set((1, 2, 43))], (None,), {"bc": "de"})
        bio = BytesIO()
        with h5py.File(bio, "w") as f:
            with self.assertWarns(UserWarning):
                res.to_nexus(f, name="test")
        with self.assertWarns(ORSOSchemaWarning):
            res = TestNxs([None, {"a": "b"}], ({"a": "b"},), set((1, 2, 3)))
        bio = BytesIO()
        with h5py.File(bio, "w") as f:
            with self.assertWarns(UserWarning):
                res.to_nexus(f, name="test")


class TestFunctions(unittest.TestCase):
    """
    Tests for functionality in the Orso module.
    """

    def test_make_empty(self):
        """
        Creation of the empty Orso object.
        """
        empty = Orso.empty()
        assert issubclass(empty.__class__, Orso)
        ds = empty.data_source
        assert ds.owner.name is None
        assert ds.experiment.title is None
        assert ds.experiment.instrument is None
        assert ds.experiment.start_date is None
        assert ds.experiment.probe is None
        assert ds.sample.name is None
        assert ds.measurement.instrument_settings.incident_angle.magnitude is None
        assert ds.measurement.instrument_settings.wavelength.magnitude is None
        assert ds.measurement.instrument_settings.polarization is Polarization.unpolarized
        assert ds.measurement.data_files == []
        assert empty.reduction.software.name is None
        assert empty.reduction.software.version is None
        assert empty.reduction.software.platform is None
        assert empty.reduction.timestamp is None
        assert empty.reduction.creator is None
        assert ds.owner.affiliation is None
        assert ds.sample.name is None
        assert empty.reduction.corrections is None
        assert empty.reduction.creator is None
        assert empty.columns == [Column("Qz", "1/angstrom"), Column(name="R")]
        assert empty.data_set is None
        dct = empty.to_dict()
        _validate_header_data([dct])

    def test_empty_to_yaml(self):
        """
        Checking yaml string form empty Orso object.

        TODO: Fix once correct format is known.
        """
        empty = Orso.empty()
        req = (
            "data_source:\n"
            "  owner:\n"
            "    name: null\n"
            "    affiliation: null\n"
            "  experiment:\n"
            "    title: null\n"
            "    instrument: null\n"
            "    start_date: null\n"
            "    probe: null\n"
            "  sample:\n"
            "    name: null\n"
            "  measurement:\n"
            "    instrument_settings:\n"
            "      incident_angle: {magnitude: null}\n"
            "      wavelength: {magnitude: null}\n"
            "      polarization: unpolarized\n"
            "    data_files: []\n"
            "reduction:\n"
            "  software: {name: null}\n"
            "columns:\n"
            "- {name: Qz, unit: 1/angstrom}\n"
            "- {name: R}\n"
        )
        assert empty.to_yaml() == req
