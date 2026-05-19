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


@dataclass
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


@dataclass
class Bilayer(Header, SubStackType):
    """
    Building block corresponding to a bilayer of lipids with outer (heads) and inner (tails)
    molecule definition. The bilayer is calculated from molecular volume, area per molecule (apm) and
    the level of hydration.
    The solvent used is either the environment set by a SubStack above or the default_solvent attribute
    of the ModelParameters.
    """

    _known_lipids = {
        "DMPC": (
            Material(formula="C10H18O8NP", number_density=Value(1.0 / 3.19, "1/nm^3")),
            Material(formula="C26H54", number_density=Value(1.0 / 7.82, "1/nm^3")),
        ),
    }

    lipid: Optional[Literal["DMPC"]] = None
    outer: Optional[Union[Material, str]] = None
    inner: Optional[Union[Material, str]] = None
    apm: Optional[Union[float, Value]] = field(default_factory=lambda: Value(0.7, unit="nm^2"))
    outer_hydration: Optional[float] = 0.3
    inner_hydration: Optional[float] = 0.3
    outer_hydration_2: Optional[float] = None
    inner_hydration_2: Optional[float] = None
    roughness: Optional[Union[float, Value]] = None
    sub_stack_class: Literal["Bilayer"] = "Bilayer"

    _environment = None

    def resolve_names(self, resolvable_items):
        if self.lipid is None:
            oi_mats = (self.outer, self.inner)
        else:
            oi_mats = self._known_lipids[self.lipid]
        self._materials = []
        for i, mi in enumerate(oi_mats):
            if isinstance(mi, Material):
                material = mi
            elif mi in resolvable_items:
                material = resolvable_items[mi]
            elif mi in SPECIAL_MATERIALS:
                material = SPECIAL_MATERIALS[mi]
            else:
                material = Material(formula=mi)
            self._materials.append(material)
        if "environment" in resolvable_items:
            self._environment = resolvable_items["environment"]
            if not isinstance(self._environment, Material):
                if self._environment in resolvable_items:
                    self._environment = resolvable_items[self._environment]
                elif self._environment in SPECIAL_MATERIALS:
                    self._environment = SPECIAL_MATERIALS[self._environment]
                elif isinstance(self._environment, str):
                    self._environment = Material(formula=self._environment)

    def resolve_defaults(self, defaults: ModelParameters) -> None:
        if not self._environment:
            self._environment = defaults.default_solvent
        else:
            self._environment.resolve_defaults(defaults)
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

    @staticmethod
    def mixed_material(material: Material, solvent: Material, fraction: float):
        srdens = solvent.number_density.magnitude / material.number_density.as_unit(solvent.number_density.unit)
        FU = f"({material.formula}){1-fraction}({solvent.formula}){srdens*fraction}"
        return Material(formula=FU, number_density=material.number_density)

    def resolve_to_layers(self) -> List[Layer]:
        for material in [self._environment] + self._materials:
            material.generate_density()
            if material.number_density is None:
                # need to generate number density from mass density and formula
                formula = Formula(material.formula, strict=True)
                fu_mass = 0.0
                for element, number in formula.elements:
                    if element.mass is None:
                        raise ValueError(f"No mass known for element {element}")
                    fu_mass += number * element.mass
                material.number_density = Value(
                    material.mass_density.as_unit("g/cm^3") / fu_mass / u2g * 1e21, unit="1/nm^3"
                )

        d_head = Value(1.0 / (self._materials[0].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm")
        d_tail = Value(1.0 / (self._materials[1].number_density.as_unit("1/nm^3") * self.apm.as_unit("nm^2")), "nm")

        m_head_1 = self.mixed_material(self._materials[0], self._environment, self.outer_hydration)
        m_head_2 = self.mixed_material(
            self._materials[0], self._environment, self.outer_hydration_2 or self.outer_hydration
        )
        m_tail_1 = self.mixed_material(self._materials[1], self._environment, self.inner_hydration)
        m_tail_2 = self.mixed_material(
            self._materials[1], self._environment, self.inner_hydration_2 or self.inner_hydration
        )

        head = Layer(thickness=d_head, material=m_head_1, roughness=self.roughness)
        tail = Layer(thickness=d_tail, material=m_tail_1, roughness=self.roughness)
        head_2 = Layer(thickness=d_tail, material=m_tail_2, roughness=self.roughness)
        tail_2 = Layer(thickness=d_head, material=m_head_2, roughness=self.roughness)
        return [head, tail, tail_2, head_2]
