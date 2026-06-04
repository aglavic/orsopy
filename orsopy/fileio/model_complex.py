"""
Build-in blocks of physical units used in model to describe more complex systems.

All these need to follow the .model_building_blocks.SubStackType protocol and
have a common "sub_stack_class" attribute that has to be set to the class name.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..slddb.constants import u2g
from ..utils.chemical_formula import Formula
from .base import Header, Literal, Value
from .model_building_blocks import SPECIAL_MATERIALS, Composit, Layer, Material, ModelParameters, SubStackType


@dataclass(repr=False)
class FunctionTwoElements(Header, SubStackType):
    """
    Models a continuous variation between two materials/SLDs according to an analytical function.

    The profile rho(z) is defined according to the relative layer thickness as fraction of material 2:
        rho(z) = (1-f((x-x0)/thickness))*rho_1 + f((x-x0)/thickness)*rho_2

    f is bracketed between 0 and 1 to prevent any artefacts with SLDs that are non-physical.

    The function string is evaluated according to python syntax using only build-in operators
    and a limited set of mathematical functions and constants defined in the class constant **ALLOWED_FUNCTIONS**.

    TODO: Review class parameters within ORSO.
    """

    material1: str
    material2: str
    function: str
    thickness: Optional[Union[float, Value]] = None
    roughness: Optional[Union[float, Value]] = None
    slice_resolution: Optional[Union[float, Value]] = None
    sub_stack_class: Literal["FunctionTwoElements"] = "FunctionTwoElements"

    ALLOWED_FUNCTIONS = [
        "pi",
        "sqrt",
        "exp",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
    ]

    def resolve_names(self, resolvable_items):
        self._materials = []
        for i, mi in enumerate([self.material1, self.material2]):
            if mi in resolvable_items:
                material = resolvable_items[mi]
            elif mi in SPECIAL_MATERIALS:
                material = SPECIAL_MATERIALS[mi]
            else:
                material = Material(formula=mi)
            self._materials.append(material)

    def resolve_defaults(self, defaults: ModelParameters) -> None:
        if self.thickness is None:
            self.thickness = Value(0.0, unit=defaults.length_unit)
        elif not isinstance(self.thickness, Value):
            self.thickness = Value(self.thickness, unit=defaults.length_unit)
        elif self.thickness.unit is None:
            self.thickness.unit = defaults.length_unit

        if self.roughness is None:
            self.roughness = defaults.roughness
        elif not isinstance(self.roughness, Value):
            self.roughness = Value(self.roughness, unit=defaults.length_unit)
        elif self.roughness.unit is None:
            self.roughness.unit = defaults.length_unit

        if self.slice_resolution is None:
            self.slice_resolution = defaults.slice_resolution
        elif not isinstance(self.slice_resolution, Value):
            self.slice_resolution = Value(self.slice_resolution, unit=defaults.length_unit)
        elif self.slice_resolution.unit is None:
            self.slice_resolution.unit = defaults.length_unit

    def resolve_to_layers(self) -> List[Layer]:
        # pre-defined math functions allowed
        glo = {}
        import math

        for name in self.ALLOWED_FUNCTIONS:
            param = getattr(math, name)
            glo[name] = param

        # use the approximate slice resolution but make sure the total thickness is exact
        length_unit = self.thickness.unit
        slices = int(round(self.thickness.magnitude / self.slice_resolution.as_unit(length_unit)))
        di = self.thickness.magnitude / slices
        thickness = Value(magnitude=di, unit=length_unit)
        roughness = Value(magnitude=di / 2.0, unit=length_unit)
        output = []
        for i in range(slices):
            loc = {"x": (i + 0.5) / slices}
            fraction = max(0.0, min(1.0, eval(self.function, glo, loc)))
            composition = Composit(composition={self.material1: (1.0 - fraction), self.material2: fraction})
            composition.resolve_names({self.material1: self._materials[0], self.material2: self._materials[1]})
            output.append(Layer(material=composition, thickness=thickness, roughness=roughness))
        output[0].roughness = self.roughness
        return output


def mix_hydrate_material(material: Material, solvent: Material, hydtration: float, coverage: float = 1.0):
    mat_frac = (1.0 - hydtration) * coverage
    srdens = solvent.number_density.magnitude / material.number_density.as_unit(solvent.number_density.unit)
    FU = f"({material.formula}){mat_frac}({solvent.formula}){srdens*(1.-mat_frac)}"
    return Material(formula=FU, number_density=material.number_density)


class LipidBase:
    """
    Common mathods used for resolving materials and defaults in Lipid based blocks.
    """

    _materials: List[Union[Composit, Material]]
    apm: Union[float, Value]
    coverage: float
    roughness: Optional[Union[float, Value]]

    def resolve_defaults(self, defaults: ModelParameters) -> None:
        if len(self._materials) != 3:
            self._materials.append(defaults.default_solvent)
        if isinstance(self.apm, Value) and self.apm.unit is None:
            self.apm.unit = defaults.length_unit + "^2"
        elif not isinstance(self.apm, Value):
            self.apm = Value(self.apm, unit=defaults.length_unit + "^2")

        for mi in self._materials:
            mi.resolve_defaults(defaults)

        if self.roughness is None:
            self.roughness = defaults.roughness
        elif not isinstance(self.roughness, Value):
            self.roughness = Value(self.roughness, unit=defaults.length_unit)
        elif self.roughness.unit is None:
            self.roughness.unit = defaults.length_unit

    def mixed_material(self, material: Material, solvent: Material, hydtration: float):
        return mix_hydrate_material(material, solvent, hydtration, self.coverage)

    def ensure_densities(self):
        for i, mi in enumerate(self._materials):
            res = mi.generate_density()
            if res is not None:
                # replace Composit by resulting Material
                self._materials[i] = res

    @property
    def solvent(self):
        return self._materials[2]

    known_lipids = {
        "DMPC": {
            "heads": Material(formula="C10H18O8NP", number_density=Value(3.135, unit="1/nm^3")),
            "tails": Material(formula="C26H54", number_density=Value(1.2, unit="1/nm^3")),
        }
    }


@dataclass(repr=False)
class Leaflet(Header, LipidBase, SubStackType):
    """
    Building block corresponding to a single layer of lipids with heads and tails
    molecule definition. The layer is calculated from molecular volume, area per molecule (apm) and
    the level of hydration.
    The solvent used is either the environment set by a SubStack above or the default_solvent attribute
    of the ModelParameters.
    The order of heads/tails with respect to the beam side can be flipped using heads_first=False.
    """

    heads: Union[Composit, Material, str]
    tails: Union[Composit, Material, str]
    heads_first: Optional[bool] = True
    apm: Optional[Union[float, Value]] = field(default_factory=lambda: Value(0.7, unit="nm^2"))
    heads_hydration: Optional[float] = 0.3
    tails_hydration: Optional[float] = 0.3
    coverage: Optional[float] = 1.0
    roughness: Optional[Union[float, Value]] = None
    sub_stack_class: Literal["Leaflet"] = "Leaflet"

    @classmethod
    def resolve_name(cls, name):
        kwds = {}
        if name.startswith("r"):
            name = name[1:]
            kwds["heads_first"] = False
        if name in cls.known_lipids:
            data = cls.known_lipids[name]
            kwds.update(data)
            return cls.from_dict(kwds)
        else:
            raise ValueError(f"Unknown lipid name: {name}")

    def resolve_names(self, resolvable_items):
        self._materials = []
        for i, mi in enumerate((self.heads, self.tails)):
            if isinstance(mi, Material):
                material = mi
            elif isinstance(mi, Composit):
                mi.resolve_names(resolvable_items)
                material = mi
            elif mi in resolvable_items:
                material = resolvable_items[mi]
                material.original_name = mi
            elif mi in SPECIAL_MATERIALS:
                material = SPECIAL_MATERIALS[mi]
                material.original_name = mi
            else:
                material = Material(formula=mi)
                material.original_name = mi
            self._materials.append(material)
        if "environment" in resolvable_items:
            environment = resolvable_items["environment"]
            if not isinstance(self._environment, (Composit, Material)):
                if environment in resolvable_items:
                    environment = resolvable_items[environment]
                elif environment in SPECIAL_MATERIALS:
                    environment = SPECIAL_MATERIALS[environment]
                elif isinstance(self._environment, str):
                    environment = Material(formula=environment)
            self._materials.append(environment)

    def resolve_to_blocks(self) -> List[Union["Layer", "SubStackType"]]:
        # Make sure the block includes full material data
        self.ensure_densities()
        self.heads = self._materials[0]
        self.tails = self._materials[1]
        return [self]

    def resolve_to_layers(self) -> List[Layer]:
        self.ensure_densities()

        Vf_head = 1.0 - self.heads_hydration
        Vf_tail = 1.0 - self.tails_hydration

        d_head = Value(
            1.0 / (Vf_head * self._materials[0].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )
        d_tail = Value(
            1.0 / (Vf_tail * self._materials[1].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )

        m_head = self.mixed_material(self._materials[0], self._materials[2], self.heads_hydration)
        m_tail = self.mixed_material(self._materials[1], self._materials[2], self.tails_hydration)

        head = Layer(thickness=d_head, material=m_head, roughness=self.roughness)
        tail = Layer(thickness=d_tail, material=m_tail, roughness=self.roughness)
        if self.heads_first:
            return [head, tail]
        else:
            return [tail, head]


@dataclass(repr=False)
class Bilayer(Header, LipidBase, SubStackType):
    """
    Building block corresponding to a bilayer of lipids with outer (heads) and inner (tails)
    molecule definition. The bilayer is calculated from molecular volume, area per molecule (apm) and
    the level of hydration.
    The solvent used is either the environment set by a SubStack above or the default_solvent attribute
    of the ModelParameters.
    """

    outer: Union[Composit, Material, str]
    inner: Union[Composit, Material, str]
    apm: Optional[Union[float, Value]] = field(default_factory=lambda: Value(0.7, unit="nm^2"))
    outer_hydration: Optional[float] = 0.3
    inner_hydration: Optional[float] = 0.3
    outer_hydration_2: Optional[float] = None
    inner_hydration_2: Optional[float] = None
    coverage: Optional[float] = 1.0
    roughness: Optional[Union[float, Value]] = None
    sub_stack_class: Literal["Bilayer"] = "Bilayer"

    @classmethod
    def resolve_name(cls, name):
        kwds = {}
        if name in cls.known_lipids:
            data = cls.known_lipids[name]
            for src, dest in [
                ("heads", "outer"),
                ("tails", "inner"),
                ("apm", "apm"),
                ("outer_hydration", "heads_hydration"),
                ("inner_hydration", "tails_hydration"),
                ("roughness", "roughness"),
            ]:
                if src in data:
                    kwds[dest] = data[src]
            return cls.from_dict(kwds)
        else:
            raise ValueError(f"Unknown lipid name: {name}")

    def resolve_names(self, resolvable_items):
        self._materials = []
        for i, mi in enumerate((self.outer, self.inner)):
            if isinstance(mi, Material):
                material = mi
            elif isinstance(mi, Composit):
                mi.resolve_names(resolvable_items)
                material = mi
            elif mi in resolvable_items:
                material = resolvable_items[mi]
                material.original_name = mi
            elif mi in SPECIAL_MATERIALS:
                material = SPECIAL_MATERIALS[mi]
                material.original_name = mi
            else:
                material = Material(formula=mi)
                material.original_name = mi
            self._materials.append(material)
        if "environment" in resolvable_items:
            environment = resolvable_items["environment"]
            if not isinstance(environment, Material):
                if environment in resolvable_items:
                    environment = resolvable_items[self._environment]
                elif environment in SPECIAL_MATERIALS:
                    environment = SPECIAL_MATERIALS[self._environment]
                elif isinstance(self._environment, str):
                    environment = Material(formula=self._environment)
            self._materials.append(environment)

    def resolve_to_blocks(self) -> List[Union["Layer", "SubStackType"]]:
        self.ensure_densities()
        # Make sure the block includes full material data
        self.inner = self._materials[1]
        self.outer = self._materials[0]
        return [self]

    def resolve_to_layers(self) -> List[Layer]:
        self.ensure_densities()

        Vf_head_1 = 1.0 - self.outer_hydration
        Vf_tail_1 = 1.0 - self.inner_hydration
        Vf_head_2 = 1.0 - (self.outer_hydration_2 or self.outer_hydration)
        Vf_tail_2 = 1.0 - (self.inner_hydration_2 or self.inner_hydration)
        d_head_1 = Value(
            1.0 / (Vf_head_1 * self._materials[0].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )
        d_head_2 = Value(
            1.0 / (Vf_head_2 * self._materials[0].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )
        d_tail_1 = Value(
            1.0 / (Vf_tail_1 * self._materials[1].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )
        d_tail_2 = Value(
            1.0 / (Vf_tail_2 * self._materials[1].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm"
        )

        m_head_1 = self.mixed_material(self._materials[0], self._materials[2], self.outer_hydration)
        m_head_2 = self.mixed_material(
            self._materials[0], self._materials[2], self.outer_hydration_2 or self.outer_hydration
        )
        m_tail_1 = self.mixed_material(self._materials[1], self._materials[2], self.inner_hydration)
        m_tail_2 = self.mixed_material(
            self._materials[1], self._materials[2], self.inner_hydration_2 or self.inner_hydration
        )

        head = Layer(thickness=d_head_1, material=m_head_1, roughness=self.roughness)
        tail = Layer(thickness=d_tail_1, material=m_tail_1, roughness=self.roughness)
        tail_2 = Layer(thickness=d_tail_2, material=m_tail_2, roughness=self.roughness)
        head_2 = Layer(thickness=d_head_2, material=m_head_2, roughness=self.roughness)
        return [head, tail, tail_2, head_2]
