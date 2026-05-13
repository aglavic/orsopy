"""
Implementation of the simplified model language for the ORSO header.

It includes parsing of models from header or different input information and
resolving the model to a simple list of slabs.
"""

import warnings

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import yaml

from ..utils.chemical_formula import Formula
from . import model_complex
from .base import Header, Literal
from .model_building_blocks import (DENSITY_RESOLVERS, SPECIAL_MATERIALS, Composit, Layer, Material, ModelParameters,
                                    SubStackType)


def find_idx(string, start, value):
    res = string[start:].find(value)
    if res >= 0:
        next_idx = start + res
    else:
        next_idx = len(string)
    return next_idx


def find_closing(string, start, brackets="()"):
    open_brackets = 1
    idx = start
    while idx < len(string):
        if string[idx] == brackets[0]:
            open_brackets += 1
        if string[idx] == brackets[1]:
            open_brackets -= 1
            if open_brackets == 0:
                return idx
        idx += 1
    return -1


@dataclass
class SubStack(Header, SubStackType):
    repetitions: int = 1
    stack: Optional[str] = None
    sequence: Optional[List[Layer]] = None
    sub_stack_class: Optional[Literal["SubStack"]] = "SubStack"
    environment: Optional[Union[str, Material, Composit]] = None

    original_name = None

    def resolve_names(self, resolvable_items):
        if isinstance(self.environment, str):
            env_orig = self.environment
            if self.environment in resolvable_items:
                self.environment = resolvable_items[self.environment]
            elif self.environment in SPECIAL_MATERIALS:
                self.environment = SPECIAL_MATERIALS[self.environment]
            else:
                self.environment = Material(formula=self.environment)
            self.environment.original_name = env_orig
        if self.environment is not None:
            resolvable_items = {"environment": self.environment, **resolvable_items}

        if self.stack is None and self.sequence is None:
            raise ValueError("SubStack has to either define stack or sequence")
        if self.sequence is None:
            stack = self.stack
            output = []
            idx = 0
            while idx < len(stack):
                next_idx = find_idx(stack, idx, "|")
                if "(" in stack[idx:next_idx]:
                    close_idx = find_closing(stack, find_idx(stack, idx, "(") + 1)
                    next_idx = find_idx(stack, close_idx, "|")
                    rep, sub_stack = stack[idx:close_idx].split("(", 1)
                    if rep.strip() == "":
                        rep = 1
                    else:
                        rep = int(rep)
                    rest = stack[close_idx + 1 : next_idx]
                    if rest.strip().startswith("in "):
                        # the Stack has elements within a matrix material
                        environment = rest.strip()[3:]
                    else:
                        # if there is a higher level envirnment, it is kept if not overwritten
                        environment = self.environment
                    obj = SubStack(repetitions=rep, stack=sub_stack.strip(), environment=environment)
                elif "{" in stack[idx:next_idx]:
                    # allow supplying SubStackType classes to be defined in the stack string directly
                    # with the syntac ClassName{p1: v1, p2: v2}
                    open_idx = find_idx(stack, idx, "{")
                    close_idx = find_closing(stack, open_idx + 1, brackets="{}")
                    class_name = stack[idx:open_idx].strip()
                    for T in SubStackType.__subclasses__():
                        if T.sub_stack_class == class_name:
                            ssclass = T
                            break
                    obj = ssclass.from_dict(yaml.load(stack[open_idx : close_idx + 1], Loader=yaml.FullLoader))
                    try:
                        thickness = float(stack[close_idx + 1 : next_idx].strip())
                    except ValueError:
                        pass
                    else:
                        if getattr(obj, "thickness", "ignore") is None:
                            obj.thickness = thickness
                else:
                    items = stack[idx:next_idx].strip().rsplit(None, 1)
                    item = items[0].strip()
                    if len(items) == 2:
                        try:
                            thickness = float(items[1])
                        except ValueError:
                            # it can't be interpreted as number, assume name has space
                            thickness = 0.0
                            item = stack[idx:next_idx].strip()
                    else:
                        thickness = 0.0

                    if item in resolvable_items:
                        obj = resolvable_items[item]
                        if isinstance(obj, SubStackType):
                            # create a copy of the object to allow different environments for same key
                            obj = obj.__class__.from_dict(obj.to_dict())
                        if isinstance(obj, Material) or isinstance(obj, Composit):
                            obj = Layer(material=obj, thickness=thickness)
                        elif getattr(obj, "thickness", "ignore") is None:
                            obj.thickness = thickness
                    else:
                        try:
                            Formula(item, strict=True)
                        except ValueError:
                            # try to resolve name directly with databse
                            res = None
                            for resolver in DENSITY_RESOLVERS:
                                res = resolver.resolve_item(item)
                                if res is not None:
                                    break
                            if res is None:
                                # assume name is a Formula to resolve within Layer
                                obj = Layer(material=item, thickness=thickness)
                                obj.original_name = item
                            else:
                                if "material" in res:
                                    obj = Layer.from_dict(res)
                                elif "composition" in res:
                                    obj = Layer(material=Composit.from_dict(res), thickness=thickness)
                                elif "formula" in res or "sld" in res:
                                    obj = Layer(material=Material.from_dict(res), thickness=thickness)
                                else:
                                    obj = Layer(material=item, thickness=thickness)
                                obj.original_name = item
                                if getattr(obj, "thickness", "ignore") is None:
                                    obj.thickness = thickness
                        else:
                            obj = Layer(material=item, thickness=thickness)
                            obj.original_name = item
                if hasattr(obj, "resolve_names"):
                    obj.resolve_names(resolvable_items)
                output.append(obj)
                idx = next_idx + 1
            self.sequence = output
        else:
            for li in self.sequence:
                li.resolve_names(resolvable_items)

    def resolve_defaults(self, defaults: ModelParameters):
        for li in self.sequence:
            if hasattr(li, "resolve_defaults"):
                li.resolve_defaults(defaults)
        if self.environment is not None:
            self.environment.resolve_defaults(defaults)

    def resolve_to_blocks(self) -> List[Union[Layer, SubStackType]]:
        # like resovle_to_layers but keeping SubStackType classes in tact
        blocks = list(self.sequence)
        added = 0
        for i in range(len(blocks)):
            if isinstance(blocks[i + added], Layer):
                if blocks[i + added].material is None:
                    # TODO: verify this case actually exists
                    blocks[i + added].generate_material()
                blocks[i + added].material.generate_density()
            else:
                obj = blocks.pop(i + added)
                sub_blocks = obj.resolve_to_blocks()
                blocks = blocks[: i + added] + sub_blocks + blocks[i + added :]
                added += len(sub_blocks) - 1
        return blocks * self.repetitions

    def resolve_to_layers(self) -> List[Layer]:
        layers = list(self.sequence)
        added = 0
        for i in range(len(layers)):
            if isinstance(layers[i + added], Layer):
                if layers[i + added].material is None:
                    layers[i + added].generate_material()
                layers[i + added].material.generate_density()
            else:
                obj = layers.pop(i + added)
                sub_layers = obj.resolve_to_layers()
                layers = layers[: i + added] + sub_layers + layers[i + added :]
                added += len(sub_layers) - 1
        return layers * self.repetitions


SUBSTACK_TYPE = SubStack
for T in SubStackType.__subclasses__():
    SUBSTACK_TYPE = Union[SUBSTACK_TYPE, T]


@dataclass
class ItemChanger(Header):
    """
    Allows to define a simple change in SubStackType item by
    just updating a selected set of parameters.
    """

    like: str
    but: dict
    original_name = None


@dataclass
class SampleModel(Header):
    stack: str
    origin: Optional[str] = None
    sub_stacks: Optional[Dict[str, Union[ItemChanger, SUBSTACK_TYPE]]] = None
    layers: Optional[Dict[str, Layer]] = None
    materials: Optional[Dict[str, Material]] = None
    composits: Optional[Dict[str, Composit]] = None
    globals: Optional[ModelParameters] = None
    reference: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        names = []
        for di in [self.sub_stacks, self.layers, self.materials, self.composits]:
            if di is None:
                continue
            for ni in di.keys():
                if ni in names:
                    warnings.warn(f'Duplicate name "{ni}" in SampleModel definition')
                names.append(ni)

    @property
    def resolvable_items(self):
        output = {}
        if self.sub_stacks:
            for key, ssi in self.sub_stacks.items():
                if isinstance(ssi, ItemChanger):
                    ssi_ref = self.sub_stacks[ssi.like]
                    ssi_ref_data = ssi_ref.to_dict()
                    ssi_ref_data.update(ssi.but)
                    ssi = ssi_ref.__class__.from_dict(ssi_ref_data)
                    self.sub_stacks[key] = ssi
                ssi.original_name = key
            output.update(self.sub_stacks)
        if self.layers:
            for key, li in self.layers.items():
                li.original_name = key
            output.update(self.layers)
        if self.materials:
            for key, mi in self.materials.items():
                mi.original_name = key
            output.update(self.materials)
        if self.composits:
            for key, ci in self.composits.items():
                ci.original_name = key
            output.update(self.composits)
        return output

    def resolve_stack(self):
        if self.globals is None:
            defaults = ModelParameters()
        else:
            defaults = self.globals
        stack = self.stack
        ri = self.resolvable_items
        output = []
        idx = 0
        while idx < len(stack):
            next_idx = find_idx(stack, idx, "|")
            if "(" in stack[idx:next_idx]:
                close_idx = find_closing(stack, find_idx(stack, idx, "(") + 1)
                next_idx = find_idx(stack, close_idx, "|")
                rep, sub_stack = stack[idx:close_idx].split("(", 1)
                if rep.strip() == "":
                    rep = 1
                else:
                    rep = int(rep)
                rest = stack[close_idx + 1 : next_idx]
                if rest.strip().startswith("in "):
                    # the Stack has elements within a matrix material
                    environment = rest.strip()[3:]
                else:
                    environment = None
                obj = SubStack(repetitions=rep, stack=sub_stack.strip(), environment=environment)
            elif "{" in stack[idx:next_idx]:
                # allow supplying SubStackType classes to be defined in the stack string directly
                # with the syntac ClassName{p1: v1, p2: v2}
                open_idx = find_idx(stack, idx, "{")
                close_idx = find_closing(stack, open_idx + 1, brackets="{}")
                class_name = stack[idx:open_idx].strip()
                for T in SubStackType.__subclasses__():
                    if T.sub_stack_class == class_name:
                        ssclass = T
                        break
                obj = ssclass.from_dict(yaml.load(stack[open_idx : close_idx + 1], Loader=yaml.FullLoader))
                try:
                    thickness = float(stack[close_idx + 1 : next_idx].strip())
                except ValueError:
                    pass
                else:
                    if getattr(obj, "thickness", "ignore") is None:
                        obj.thickness = thickness
            else:
                items = stack[idx:next_idx].strip().rsplit(None, 1)
                item = items[0].strip()
                if len(items) == 2:
                    try:
                        thickness = float(items[1])
                    except ValueError:
                        # if can't be interpreted as umber, assume name has space
                        thickness = 0.0
                        item = stack[idx:next_idx].strip()
                else:
                    thickness = 0.0

                if item in ri:
                    obj = ri[item]
                    if isinstance(obj, Material) or isinstance(obj, Composit):
                        obj = Layer(material=obj, thickness=thickness)
                    elif getattr(obj, "thickness", "ignore") is None:
                        obj.thickness = thickness
                else:
                    try:
                        Formula(item, strict=True)
                    except ValueError:
                        # try to resolve name directly with databse
                        res = None
                        if len(DENSITY_RESOLVERS) == 0:
                            from ..utils.resolver_slddb import ResolverSLDDB

                            DENSITY_RESOLVERS.append(ResolverSLDDB())
                        for resolver in DENSITY_RESOLVERS:
                            res = resolver.resolve_item(item)
                            if res is not None:
                                break
                        if res is None:
                            # assume name is a Formula to resolve within Layer
                            obj = Layer(material=item, thickness=thickness)
                            obj.original_name = item
                        else:
                            if "material" in res:
                                obj = Layer.from_dict(res)
                            elif "composition" in res:
                                obj = Layer(material=Composit.from_dict(res), thickness=thickness)
                            elif "formula" in res or "sld" in res:
                                obj = Layer(material=Material.from_dict(res), thickness=thickness)
                            else:
                                obj = Layer(material=item, thickness=thickness)
                            obj.original_name = item
                            if getattr(obj, "thickness", "ignore") is None:
                                obj.thickness = thickness
                    else:
                        obj = Layer(material=item, thickness=thickness)
                        obj.original_name = item
            if hasattr(obj, "resolve_names"):
                obj.resolve_names(ri)
            if hasattr(obj, "resolve_defaults"):
                obj.resolve_defaults(defaults)
            output.append(obj)
            idx = next_idx + 1
        return output

    def resolve_to_blocks(self) -> List[Union[Layer, SubStackType]]:
        # like resovle_to_layers but keeping SubStackType classes intact
        blocks = self.resolve_stack()
        added = 0
        for i in range(len(blocks)):
            if isinstance(blocks[i + added], Layer):
                if blocks[i + added].material is None:
                    blocks[i + added].generate_material()
                blocks[i + added].material.generate_density()
            else:
                obj = blocks.pop(i + added)
                sub_blocks = obj.resolve_to_blocks()
                blocks = blocks[: i + added] + sub_blocks + blocks[i + added :]
                added += len(sub_blocks) - 1
        return blocks

    def resolve_to_layers(self) -> List[Layer]:
        layers = self.resolve_stack()
        added = 0
        for i in range(len(layers)):
            if isinstance(layers[i + added], Layer):
                if layers[i + added].material is None:
                    layers[i + added].generate_material()
                layers[i + added].material.generate_density()
            else:
                obj = layers.pop(i + added)
                sub_layers = obj.resolve_to_layers()
                layers = layers[: i + added] + sub_layers + layers[i + added :]
                added += len(sub_layers) - 1
        return layers
