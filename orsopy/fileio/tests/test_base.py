"""
Tests for fileio.base module
"""

# pylint: disable=R0201

import datetime as datetime_module
import sys
import unittest

from dataclasses import dataclass
from datetime import datetime
from math import log, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pint
import pytest

from orsopy.fileio import base, orso

pth = Path(__file__).absolute().parent


class TestHeaderClass(unittest.TestCase):
    """
    Testing general class functionalities of ORSO Header.
    """

    def test_resolve_any(self):
        @dataclass(repr=False)
        class TestAny(base.Header):
            test: Any

        test_object = [1, 2, "test"]
        res = TestAny(test=test_object)
        assert res.test is test_object

    def test_resolve_datetime(self):
        @dataclass(repr=False)
        class TestDatetime(base.Header):
            test: datetime

        res = TestDatetime(test=datetime.now())
        assert type(res.test) is datetime
        res = TestDatetime(test=datetime.now().isoformat("T"))
        assert type(res.test) is datetime
        with self.assertWarns(base.ORSOSchemaWarning):
            res = TestDatetime(test="no-ISO-string")
        assert type(res.test) is str

        # check python 3.6 implementation where fromisoformat did not exist
        class MockDatetime:
            @staticmethod
            def strptime(item, format):
                return datetime.strptime(item, format)

        @dataclass(repr=False)
        class TestDatetime(base.Header):
            test: MockDatetime

        try:
            datetime_module.datetime = MockDatetime
            res = TestDatetime(test=MockDatetime())
            assert type(res.test) is MockDatetime
            res = TestDatetime(test=datetime.now().isoformat("T"))
            assert type(res.test) is datetime
            res = TestDatetime(test=datetime.now().strftime("%Y-%m-%d"))
            assert type(res.test) is datetime
            res = TestDatetime(test=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
            assert type(res.test) is datetime
            with self.assertWarns(base.ORSOSchemaWarning):
                res = TestDatetime(test="no-ISO-string")
            assert type(res.test) is str
        finally:
            datetime_module.datetime = datetime

    def test_resolve_dictof(self):
        if sys.version_info < (3, 8):
            # dict type annotation changed in 3.8
            return

        @dataclass(repr=False)
        class TestDictof(base.Header):
            test: Dict[str, datetime]
            test2: Dict[float, int]

        now = datetime.now()
        res = TestDictof(test={"ab": now, "def": now.isoformat("T")}, test2={13: 4.0, "23": 23})
        assert res.test == {"ab": now, "def": now}
        assert res.test2 == {13.0: 4, 23.0: 23}
        assert [(type(ki), type(vi)) for ki, vi in res.test2.items()] == [(float, int), (float, int)]

        # Dict should be used to define key/value types
        @dataclass(repr=False)
        class TestDictNodef(base.Header):
            test: Dict

        with self.assertWarns(base.ORSOSchemaWarning):
            res = TestDictNodef(test={"ab": "cd"})
        assert res.test == {"ab": "cd"}

    def test_resolve_list(self):
        @dataclass(repr=False)
        class TestList(base.Header):
            test: List[int]

        res = TestList(test=[1, 2, 3])
        assert res.test == [1, 2, 3]
        res = TestList(test=["1", 2.4, 3.232])
        assert res.test == [1, 2, 3]
        res = TestList(test=("1", 2.4, 3.232))
        assert res.test == [1, 2, 3]
        res = TestList(test="134")
        assert res.test == [134]

    def test_resolve_tuple(self):
        @dataclass(repr=False)
        class TestTuple(base.Header):
            test: Tuple[int, int, int]

        res = TestTuple(test=(1, 2, 3))
        assert res.test == (1, 2, 3)
        res = TestTuple(test=["1", 2.4, 3.232])
        assert res.test == (1, 2, 3)
        res = TestTuple(test="134")
        assert res.test == (134,)
        # wrong number of elements will be shortened to type hint length
        with self.assertWarns(base.ORSOSchemaWarning):
            res = TestTuple(test=(1, 2))
        assert res.test == (1, 2)
        with self.assertWarns(base.ORSOSchemaWarning):
            res = TestTuple(test=(1, 2, 3, 4))
        assert res.test == (1, 2, 3)

    def test_subsubclass(self):
        @dataclass(repr=False)
        class Test1(base.Header):
            pass

        with self.assertRaises(NotImplementedError):

            @dataclass(repr=False)
            class Test2(Test1):
                pass

    def test_unexpected_error(self):
        """
        Test that any unforeseen exception in the resolve function gets
        reported as ORSOResolveError.
        """

        @dataclass(repr=False)
        class TestDatetime(base.Header):
            test: datetime

        def resolve_type(hint: type, item: Any):
            raise RuntimeError("Test error")

        TestDatetime._resolve_type = resolve_type
        with self.assertRaises(base.ORSOResolveError):
            TestDatetime(test=datetime.now().isoformat())

    def test_empty_creation(self):
        """
        Make a class that tests all cases of the empty method.
        """

        @dataclass(repr=False)
        class TestEmpty(base.Header):
            test1: float
            test2: base.Value
            test3: Union[base.Value, base.ErrorValue]
            test4: List[base.Value]
            test5: List[int]
            test6: Optional[float] = None

        res = TestEmpty.empty()
        assert res.test1 is None
        assert res.test2 == base.Value.empty()
        assert res.test3 == base.Value.empty()
        assert res.test4 == [base.Value.empty()]
        assert res.test5 == []
        assert res.test6 is None
        assert res.asdict(res) == res.to_dict()

    def test_dict_conversion(self):
        res = base._todict({"a": "b"})
        assert res == {"a": "b"}
        res = base._todict([1, 2, 3, 4])
        assert res == [1, 2, 3, 4]

        class Test1:
            def _ast(self):
                return {"a": "b"}

        res = base._todict(Test1())
        assert res == {"a": "b"}

        @dataclass(repr=False)
        class Test2:
            test: int
            test2: float
            test3: str

        res = base._todict(Test2(13, 12.4, "1234"), classkey="TestClassKey")
        assert res == {"test": 13, "test2": 12.4, "test3": "1234", "TestClassKey": "Test2"}

    def test_empty_optional(self):
        @dataclass(repr=False)
        class TestOptional(base.Header):
            test1: str
            test2: Optional[float] = None

        t1 = TestOptional(test1="abc", test2=13.4)
        t2 = TestOptional(test1="abc")
        # Make sure the empty optional is not present in export
        self.assertEqual(t1.to_dict(), {"test1": "abc", "test2": 13.4})
        self.assertEqual(t2.to_dict(), {"test1": "abc"})
        self.assertEqual(t1.to_yaml().strip(), "test1: abc\ntest2: 13.4")
        self.assertEqual(t2.to_yaml().strip(), "test1: abc")

    def test_default_optional(self):
        @dataclass(repr=False)
        class TestOptional(base.Header):
            test1: str
            test2: Optional[float] = 14.5

        t1 = TestOptional(test1="abc", test2=13.4)
        t2 = TestOptional(test1="abc")
        # test that default value actually is set on object
        self.assertEqual(t2.test2, 14.5)
        # Make sure the default optional is not present in export
        self.assertEqual(t1.to_dict(), {"test1": "abc", "test2": 13.4})
        self.assertEqual(t2.to_dict(), {"test1": "abc"})
        self.assertEqual(t1.to_yaml().strip(), "test1: abc\ntest2: 13.4")
        self.assertEqual(t2.to_yaml().strip(), "test1: abc")


class TestErrorValue(unittest.TestCase):
    """
    Testing the Value class.
    """

    def test_single_value(self):
        """
        Creation of an object with a magnitude and unit.
        """
        error = base.ErrorValue(1.0)
        assert error.error_value == 1.0

    def test_optionals(self):
        """
        Creation of an object with a magnitude and unit.
        """
        error = base.ErrorValue(1.0, "resolution", "FWHM", "uniform")
        assert error.error_value == 1.0
        assert error.error_type == "resolution"
        assert error.value_is == "FWHM"
        assert error.distribution == "uniform"

    def test_sigma(self):
        """
        Creation of an object with a magnitude and unit.
        """
        error = base.ErrorValue(1.0)
        assert error.sigma == 1.0
        for dist in ["gaussian", "triangular", "uniform", "lorentzian"]:
            error = base.ErrorValue(1.0, "resolution", "sigma", dist)
            assert error.sigma == 1.0
        error = base.ErrorValue(1.0, "resolution", "FWHM", "gaussian")
        assert error.sigma == 1.0 / (2.0 * sqrt(2.0 * log(2.0)))
        error = base.ErrorValue(1.0, "resolution", "FWHM", "triangular")
        assert error.sigma == 1.0 / sqrt(6)
        error = base.ErrorValue(1.0, "resolution", "FWHM", "uniform")
        assert error.sigma == 1.0 / sqrt(12)

        error = base.ErrorValue(1.0, "resolution", "FWHM", "lorentzian")
        with self.assertRaises(ValueError):
            error.sigma

        with self.assertWarns(base.ORSOSchemaWarning):
            error = base.ErrorValue(1.0, "resolution", "FWHM", "undefined")
        with self.assertRaises(NotImplementedError):
            error.sigma

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        error = base.ErrorValue(1.0)
        assert error.to_yaml() == "{error_value: 1.0}\n"

    def test_no_value_to_yaml(self):
        """
        Transform to yaml with a non-optional ORSO item.
        """
        error = base.ErrorValue(None)
        assert error.to_yaml() == "{error_value: null}\n"

    def test_optionals_to_yaml(self):
        """
        Transform to yaml.
        """
        error = base.ErrorValue(1.0)
        assert error.to_yaml() == "{error_value: 1.0}\n"

    def test_user_data(self):
        error = base.ErrorValue.from_dict(dict(error_value=13.4, my_attr="hallo ORSO"))
        assert error.user_data == {"my_attr": "hallo ORSO"}


class TestValue(unittest.TestCase):
    """
    Testing the Value class.
    """

    def test_single_value(self):
        """
        Creation of an object with a magnitude and unit.
        """
        value = base.Value(1.0, "m")
        assert value.magnitude == 1.0
        assert value.unit == "m"

    def test_list_warning(self):
        """
        Creation of an object with a a list of values and a unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            base.Value([1, 2, 3], "m")

    def test_bad_unit(self):
        """
        Rejection of non-ASCII units.
        """
        with self.assertRaises(ValueError):
            _ = base.Value(1.0, "Å")

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.Value(1.0, "m")
        assert value.to_yaml() == "{magnitude: 1.0, unit: m}\n"

    def test_no_magnitude_to_yaml(self):
        """
        Transform to yaml with a non-optional ORSO item.
        """
        value = base.Value(None)
        assert value.to_yaml() == "{magnitude: null}\n"

    def test_user_data(self):
        value = base.Value.from_dict(dict(magnitude=13.4, my_attr="hallo ORSO"))
        assert value.user_data == {"my_attr": "hallo ORSO"}

    def test_unit_conversion(self):
        base.unit_registry = None
        value = base.Value(1.0, "mm")
        assert value.as_unit("m") == 1.0e-3
        value = base.Value(1.0, "1/nm^3")
        assert value.as_unit("1/angstrom^3") == 1.0e-3

        with self.assertRaises(pint.DimensionalityError):
            value = base.Value(1.0, "1/nm^3")
            value.as_unit("m")


class TestComplexValue(unittest.TestCase):
    """
    Testing the Value class.
    """

    def test_single_value(self):
        """
        Creation of an object with a magnitude and unit.
        """
        value = base.ComplexValue(1.0, 2.0, "m")
        assert value.real == 1.0
        assert value.imag == 2.0
        assert value.unit == "m"
        repr(value)  # just test that the custom repr method does not through an exception

    def test_list_warning(self):
        """
        Creation of an object with a list of values and a unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            base.ComplexValue([1, 2, 3], [1, 2, 3], "m")

    def test_bad_unit(self):
        """
        Rejection of non-ASCII units.
        """
        with self.assertRaises(ValueError):
            _ = base.ComplexValue(1.0, 2.0, "Å")

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.ComplexValue(1.0, 2.0, "m")
        assert value.to_yaml() == "{real: 1.0, imag: 2.0, unit: m}\n"

    def test_no_magnitude_to_yaml(self):
        """
        Transform to yaml with a non-optional ORSO item.
        """
        value = base.ComplexValue(None)
        assert value.to_yaml() == "{real: null}\n"

    def test_unit_conversion(self):
        base.unit_registry = None
        value = base.ComplexValue(1.0, 2.0, "mm")
        assert value.as_unit("m") == 1.0e-3 + 2.0e-3j
        value = base.ComplexValue(1.0, 2.0, "1/nm^3")
        assert value.as_unit("1/angstrom^3") == 1.0e-3 + 2.0e-3j
        value = base.ComplexValue(1.0, None, "mm")
        assert value.as_unit("m") == 1.0e-3 + 0.0j

        with self.assertRaises(pint.DimensionalityError):
            value = base.ComplexValue(1.0, 2.0, "1/nm^3")
            value.as_unit("m")


class TestValueVector(unittest.TestCase):
    """
    Testing the ValueVector class
    """

    def test_single_value(self):
        """
        Creation of an object with three dimensions and unit.
        """
        value = base.ValueVector(1, 2, 3, "m")
        assert value.x == 1
        assert value.y == 2
        assert value.z == 3
        assert value.unit == "m"

    def test_list_warns(self):
        """
        Creation of an object with three dimensions of lists and unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            base.ValueVector([1, 2], [2, 3], [3, 4], "m")

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.ValueVector(1, 2, 3, "Å")

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.ValueVector(1.0, 2.0, 3.0, "m")
        assert value.to_yaml() == "{x: 1.0, y: 2.0, z: 3.0, unit: m}\n"

    def test_two_dimensional_to_yaml(self):
        """
        Transform to yaml with only two dimensions.
        """
        value = base.ValueVector(1.0, 2.0, None, "m")
        assert value.to_yaml() == "{x: 1.0, y: 2.0, z: null, unit: m}\n"

    def test_unit_conversion(self):
        base.unit_registry = None
        value = base.ValueVector(1.0, 2.0, 3.0, "mm")
        assert value.as_unit("mm") == (1.0, 2.0, 3.0)
        assert value.as_unit("m") == (1.0e-3, 2.0e-3, 3.0e-3)
        value = base.ValueVector(1.0, 2.0, 3.0, "1/nm^3")
        assert value.as_unit("1/angstrom^3") == (1.0e-3, 2.0e-3, 3.0e-3)

        with self.assertRaises(pint.DimensionalityError):
            value = base.ValueVector(1.0, 2.0, 3.0, "1/m")
            value.as_unit("m")


class TestValueRange(unittest.TestCase):
    """
    Testing the ValueRange class
    """

    def test_single_value(self):
        """
        Creation of an object with a max, min and unit.
        """
        value = base.ValueRange(1.0, 2.0, "m")
        assert value.min == 1
        assert value.max == 2
        assert value.unit == "m"

    def test_list_warns(self):
        """
        Creation of an object of a list of max and list of min and a unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            base.ValueRange([1, 2, 3], [2, 3, 4], "m")

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.ValueRange(1.0, 2.0, "Å")

    def test_to_yaml(self):
        """
        Transform to yaml.
        """
        value = base.ValueRange(1.0, 2.0, "m")
        assert value.to_yaml() == "{min: 1.0, max: 2.0, unit: m}\n"

    def test_no_upper_to_yaml(self):
        """
        Transform to yaml with no max.
        """
        value = base.ValueRange(1.0, None)
        assert value.to_yaml() == "{min: 1.0, max: null}\n"

    def test_no_lower_to_yaml(self):
        """
        Transform to yaml with no min.
        """
        value = base.ValueRange(None, 1.0)
        assert value.to_yaml() == "{min: null, max: 1.0}\n"

    def test_unit_conversion(self):
        base.unit_registry = None
        value = base.ValueRange(1.0, 2.0, "mm")
        assert value.as_unit("mm") == (1.0, 2.0)
        assert value.as_unit("m") == (1.0e-3, 2.0e-3)
        value = base.ValueRange(1.0, 2.0, "1/nm^3")
        assert value.as_unit("1/angstrom^3") == (1.0e-3, 2.0e-3)

        with self.assertRaises(pint.DimensionalityError):
            value = base.ValueRange(1.0, 2.0, "1/m")
            value.as_unit("m")


class TestPerson(unittest.TestCase):
    """
    Testing the Person class
    """

    def test_creation(self):
        """
        Creation with no email.
        """
        value = base.Person("Joe A. User", "Ivy League University")
        assert value.name == "Joe A. User"
        assert value.affiliation == "Ivy League University"

    def test_creation_with_contact(self):
        """
        Creation with an email.
        """
        value = base.Person("Joe A. User", "Ivy League University", "jauser@ivy.edu")
        assert value.name == "Joe A. User"
        assert value.affiliation == "Ivy League University"
        assert value.contact == "jauser@ivy.edu"

    def test_creation_with_multiline(self):
        value = base.Person("Joe A. User", "\n".join(["Ivy League University", "Great Neutron Factory"]))
        assert value.name == "Joe A. User"
        assert value.affiliation == "\n".join(["Ivy League University", "Great Neutron Factory"])

    def test_to_yaml(self):
        """
        Transform to yaml with no email.
        """
        value = base.Person("Joe A. User", "Ivy League University")
        assert value.to_yaml() == "name: Joe A. User\naffiliation: Ivy League University\n"

    def test_no_affiliation_to_yaml(self):
        """
        Transform to yaml without affiliation.
        """
        value = base.Person("Joe A. User", None)
        assert value.to_yaml() == "name: Joe A. User\naffiliation: null\n"

    def test_no_name_to_yaml(self):
        """
        Transform to yaml without name.
        """
        value = base.Person(None, "A University")
        assert value.to_yaml() == "name: null\naffiliation: A University\n"

    def test_email_to_yaml(self):
        """
        Transform to yaml with an email.
        """
        value = base.Person("Joe A. User", "Ivy League University", "jauser@ivy.edu")
        assert (
            value.to_yaml() == "name: Joe A. User\naffiliation: Ivy League University" + "\ncontact: jauser@ivy.edu\n"
        )


class TestColumn(unittest.TestCase):
    """
    Testing the Column class
    """

    def test_creation(self):
        """
        Creation of a column.
        """
        value = base.Column("q", "1/angstrom", "qz vector")
        assert value.physical_quantity == "qz vector"
        assert value.name == "q"
        assert value.unit == "1/angstrom"

    def test_bad_unit(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertRaises(ValueError):
            _ = base.Column("q", "Å", "qz vector")

    def test_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.Column("q", "1/angstrom", "qz vector")
        assert value.to_yaml() == "{name: q, unit: 1/angstrom, physical_quantity: qz vector}\n"

    def test_no_description_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.Column("q", "1/angstrom")
        assert value.to_yaml() == "{name: q, unit: 1/angstrom}\n"


class TestErrorColumn(unittest.TestCase):
    """
    Testing the Column class
    """

    def test_creation(self):
        """
        Creation of a column.
        """
        value = base.ErrorColumn("q", "uncertainty", "FWHM", "triangular")
        assert value.error_of == "q"
        assert value.error_type == "uncertainty"
        assert value.distribution == "triangular"
        assert value.value_is == "FWHM"
        value = base.ErrorColumn("q", "resolution", "sigma", "gaussian")
        assert value.error_of == "q"
        assert value.error_type == "resolution"
        assert value.distribution == "gaussian"
        assert value.value_is == "sigma"
        assert value.name == "sq"

    def test_bad_type(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            _ = base.ErrorColumn("q", "nm")

    def test_bad_distribution(self):
        """
        Rejection of non-ASCII unit.
        """
        with self.assertWarns(base.ORSOSchemaWarning):
            _ = base.ErrorColumn("q", "uncertainty", "FWHM", "undefined")
        with self.assertWarns(base.ORSOSchemaWarning):
            _ = base.ErrorColumn("q", "uncertainty", "HWHM", "triangular")
        with self.assertWarns(base.ORSOSchemaWarning):
            _ = base.ErrorColumn("q", "uncertainty", "wrong")

    def test_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.ErrorColumn("q", "uncertainty", "FWHM", "triangular")
        assert value.to_yaml() == "{error_of: q, error_type: uncertainty, value_is: FWHM, distribution: triangular}\n"

    def test_minimal_to_yaml(self):
        """
        Transformation to yaml.
        """
        value = base.ErrorColumn("q")
        assert value.to_yaml() == "{error_of: q}\n"

    def test_sigma_conversion(self):
        with self.subTest("noop"):
            value = base.ErrorColumn("q", "uncertainty", "sigma", "gaussian")
            self.assertEqual(value.to_sigma, 1.0)
        with self.subTest("gauss"):
            value = base.ErrorColumn("q", "uncertainty", "FWHM", "gaussian")
            self.assertEqual(value.to_sigma, 1.0 / (2.0 * sqrt(2.0 * log(2.0))))
        with self.subTest("uniform"):
            value = base.ErrorColumn("q", "uncertainty", "FWHM", "uniform")
            self.assertEqual(value.to_sigma, 1.0 / sqrt(12.0))
        with self.subTest("triangular"):
            value = base.ErrorColumn("q", "uncertainty", "FWHM", "triangular")
            self.assertEqual(value.to_sigma, 1.0 / sqrt(6.0))
        with self.subTest("lorentizan"):
            value = base.ErrorColumn("q", "uncertainty", "FWHM", "lorentzian")
            with self.assertRaises(ValueError):
                value.to_sigma
        with self.subTest("unknown"):
            with self.assertWarns(base.ORSOSchemaWarning):
                value = base.ErrorColumn("q", "uncertainty", "FWHM", "unknown")
            with self.assertRaises(NotImplementedError):
                value.to_sigma


class TestFile(unittest.TestCase):
    """
    Testing the File class.
    """

    def test_creation_for_nonexistent_file(self):
        """
        Creation of a file that does not exist.
        """
        value = base.File("not_a_file.txt", datetime(2021, 7, 12, 14, 4, 20))
        assert value.file == "not_a_file.txt"
        assert value.timestamp == datetime(2021, 7, 12, 14, 4, 20)

    def test_to_yaml_for_nonexistent_file(self):
        """
        Transformation to yaml of a file that does not exist.
        """
        value = base.File("not_a_file.txt", datetime(2021, 7, 12, 14, 4, 20))
        assert value.to_yaml() == "file: not_a_file.txt\ntimestamp: " + "2021-07-12T14:04:20\n"

    def test_creation_for_existing_file(self):
        """
        Creation for a file that does exist with a given modified date.
        """
        fname = Path(pth / "not_orso.ort")
        value = base.File(str(fname.absolute()), datetime.fromtimestamp(fname.stat().st_mtime))
        assert value.file == str(fname)
        assert value.timestamp == datetime.fromtimestamp(fname.stat().st_mtime)

    def test_to_yaml_for_existing_file(self):
        """
        Transformation to yaml a file that does exist with a given modified
        date.
        """
        fname = Path(pth / "not_orso.ort")
        value = base.File(str(fname.absolute()), datetime.fromtimestamp(fname.stat().st_mtime))
        assert (
            value.to_yaml()
            == f"file: {str(fname)}\n"
            + "timestamp: "
            + f"{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n"
        )

    def test_creation_for_existing_file_no_mod_time(self):
        """
        Transformation to yaml a file that does exist without a given
        modified date.
        """
        fname = Path(pth / "not_orso.ort")
        value = base.File(str(fname.absolute()), None)
        assert value.file == str(fname)
        assert value.timestamp == datetime.fromtimestamp(fname.stat().st_mtime)

    def test_to_yaml_for_existing_file_no_mod_time(self):
        """
        Transformation to yaml a file that does exist without a given
        modified date.
        """
        fname = Path(pth / "not_orso.ort")
        value = base.File(str(fname.absolute()), None)
        assert (
            value.to_yaml()
            == "file: "
            + f"{str(fname)}\n"
            + "timestamp: "
            + f"{datetime.fromtimestamp(fname.stat().st_mtime).isoformat()}\n"
        )

    def test_not_orso(self):
        with pytest.raises(base.NotOrsoCompatibleFileError, match="First line does not appear"):
            with open(pth / "not_orso.ort", "r") as f:
                orso.load_orso(f)

    def test_hashdigest(self):
        """
        Test creation of ContentHash from this python file and test if re-creating from dictionary works.
        """
        file = base.File(__file__, datetime.now())
        file.hash = base.ContentHash.from_file(__file__)
        refile = base.File.from_dict(file.to_dict())
        assert refile.hash.check_file(__file__)
