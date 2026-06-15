"""
Python script to showcase the simple model language by resolving a stack string
and plotting SLD and reflectivity for neutrons and x-rays.
The refnx model is first generate as a string and then evaluated to be able to export
the model directly.

Script requires matplotlib and refnx to run:
   pip install matplotlib refnx

If ran without installing orsopy, remember to put folder in PYTHONPATH.
Be aware of other orsopy version that might be installed by pip.
To be sure run: pip uninstall orsopy
"""

import sys

import yaml

from matplotlib import pyplot
from numpy import linspace

from orsopy.fileio import model_complex, model_language


class RefNxResolver:
    def __init__(self, sample: model_language.SampleModel):
        self.sample = sample

    def get_model(self, xray_energy=None, layers_only=False):
        self.xray_energy = xray_energy

        self.model_header()
        if layers_only:
            self.resolve_to_layers(self.sample)
        else:
            for block in self.sample.resolve_to_blocks():
                block_resolver = getattr(self, "res" + type(block).__name__, self.resolve_to_layers)
                block_resolver(block)
        self.model_end()
        return self.model

    def model_header(self):
        self.model = "from refnx.reflect import SLD, LipidLeaflet, ReflectModel, Slab, Structure\n\n"
        self.model += "structure = Structure()\n\n"

    def model_end(self):
        self.model += "model = ReflectModel(structure, bkg=0.0)\n"

    def resolve_to_layers(self, block: model_language.SubStackType):
        for li in block.resolve_to_layers():
            self.resLayer(li)

    def resLayer(self, layer: model_language.Layer):
        self.model += f"m = SLD({layer.material.get_sld(xray_energy=self.xray_energy) * 1e6})\n"
        self.model += (
            f"structure |= Slab({layer.thickness.as_unit('angstrom')}, m, "
            f"{layer.roughness.as_unit('angstrom')}, "
            f"name='{getattr(layer, 'original_name', '')}')\n\n"
        )

    def resLeaflet(self, leaflet: model_complex.Leaflet):
        apm = leaflet.apm.as_unit("angstrom^2")
        vm_heads = leaflet.heads.volume.as_unit("angstrom^3")
        vfrac = 1.0 - leaflet.hydration
        b_heads = leaflet.heads.get_sld(xray_energy=self.xray_energy) * vm_heads
        d_heads = vm_heads / apm / vfrac
        vm_tails = leaflet.tails.volume.as_unit("angstrom^3")
        b_tails = leaflet.tails.get_sld(xray_energy=self.xray_energy) * vm_tails
        d_tails = vm_tails / apm / vfrac
        solvent = leaflet.solvent.get_sld(xray_energy=self.xray_energy) * 1e6
        self.model += f"""structure |= LipidLeaflet(
            apm={apm / leaflet.coverage},
            b_heads={b_heads},
            vm_heads={vm_heads},
            thickness_heads={d_heads},
            b_tails={b_tails},
            vm_tails={vm_tails},
            thickness_tails={d_tails},
            rough_head_tail={leaflet.roughness.as_unit("angstrom")},
            rough_preceding_mono={leaflet.roughness.as_unit("angstrom")},
            reverse_monolayer={not leaflet.heads_first},
            name='{'LL ' + getattr(leaflet, 'original_name', '')}',
            head_solvent={solvent},
            tail_solvent={solvent},
        )\n\n"""

    def resBilayer(self, bilayer: model_complex.Bilayer):
        apm = bilayer.apm.as_unit("angstrom^2")
        vm_heads = bilayer.heads.volume.as_unit("angstrom^3")
        vfrac = 1.0 - bilayer.hydration
        b_heads = bilayer.heads.get_sld(xray_energy=self.xray_energy) * vm_heads
        d_heads = vm_heads / apm / vfrac
        vm_tails = bilayer.tails.volume.as_unit("angstrom^3")
        b_tails = bilayer.tails.get_sld(xray_energy=self.xray_energy) * vm_tails
        d_tails = vm_tails / apm / vfrac
        solvent = bilayer.solvent.get_sld(xray_energy=self.xray_energy) * 1e6
        self.model += f"""structure |= LipidLeaflet(
            apm={apm / bilayer.coverage},
            b_heads={b_heads},
            vm_heads={vm_heads},
            thickness_heads={d_heads},
            b_tails={b_tails},
            vm_tails={vm_tails},
            thickness_tails={d_tails},
            rough_head_tail={bilayer.roughness.as_unit("angstrom")},
            rough_preceding_mono={bilayer.roughness.as_unit("angstrom")},
            reverse_monolayer=False,
            name='{'LL ' + getattr(bilayer, 'original_name', '')}',
            head_solvent={solvent},
            tail_solvent={solvent},
        )\n"""
        vfrac = 1.0 - (bilayer.hydration_2 or bilayer.hydration)
        d_heads = vm_heads / apm / vfrac
        d_tails = vm_tails / apm / vfrac
        self.model += f"""structure |= LipidLeaflet(
            apm={apm / bilayer.coverage},
            b_heads={b_heads},
            vm_heads={vm_heads},
            thickness_heads={d_heads},
            b_tails={b_tails},
            vm_tails={vm_tails},
            thickness_tails={d_tails},
            rough_head_tail={bilayer.roughness.as_unit("angstrom")},
            rough_preceding_mono={bilayer.roughness.as_unit("angstrom")},
            reverse_monolayer=True,
            name='{'fLL ' + getattr(bilayer, 'original_name', '')}',
            head_solvent={solvent},
            tail_solvent={solvent},
        )\n\n"""


q = linspace(0.001, 0.2, 200)


def main(txt=None):
    if txt is None:
        txt = sys.argv[1]
    if txt.endswith(".yml"):
        dtxt = yaml.safe_load(open(txt, "r").read())
        if "data_source" in dtxt:
            dtxt = dtxt["data_source"]
        if "sample" in dtxt:
            dtxt = dtxt["sample"]["model"]
        sample = model_language.SampleModel.from_dict(dtxt)
        txt += f"\n{sample.stack}"
    else:
        sample = model_language.SampleModel(stack=txt)
    # initial model before resolving any names
    print(repr(sample), "\n")
    print("\n".join([repr(ss) for ss in sample.resolve_stack()]), "\n")

    blocks = sample.resolve_to_blocks()
    print("\n".join([repr(ss) for ss in blocks]), "\n")

    resolver = RefNxResolver(sample)
    model_n = resolver.get_model()
    model_x = resolver.get_model(xray_energy="Cu")

    layers = sample.resolve_to_layers()
    print("\n".join([repr(li) for li in layers]))

    # high-level resolution based on blocks
    resn = {}
    exec(model_n, resn)
    resx = {}
    exec(model_x, resx)

    print("####################### Neutron Model ##################")
    print(model_n)

    print("\n\n####################### X-ray Model ##################")
    print(model_n)

    # ORSOpy slab resolution
    model = resolver.get_model(layers_only=True)
    res = {}
    exec(model, res)
    orsopy_neutron = res["structure"]
    model = resolver.get_model(xray_energy="Cu", layers_only=True)
    res = {}
    exec(model, res)
    orsopy_xray = res["structure"]

    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(121)
    pyplot.semilogy(q, resn["model"](q), label="neutron")
    pyplot.semilogy(q, resx["model"](q), label="x-ray (Cu)")
    pyplot.legend()
    pyplot.title(txt)
    pyplot.xlabel("q [Å$^{-1}$]")
    pyplot.ylabel("Neutron-reflectivity")
    pyplot.subplot(122)
    pyplot.plot(*resn["structure"].sld_profile(), label="neutron", color="C0")
    pyplot.plot(*orsopy_neutron.sld_profile(), "--", lw=2, label="", color="C0")
    pyplot.plot(*resx["structure"].sld_profile(), label="x-ray (Cu)", color="C1")
    pyplot.plot(*orsopy_xray.sld_profile(), "--", lw=2, label="", color="C1")
    pyplot.legend()
    pyplot.title(txt)
    pyplot.ylabel("SLD / $10^{-6} \\AA^{-2}$")
    pyplot.xlabel("distance / $\\AA$")
    pyplot.show()


if __name__ == "__main__":
    main()
