"""
Python script to showcase the simple model language by resolving a stack string
and plotting SLD and reflectivity for neutrons.

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
from refnx.reflect import SLD, LipidLeaflet, ReflectModel, Slab, Structure

from orsopy.fileio import model_complex, model_language

q = linspace(0.001, 0.2, 200)


def res_layer(layer: model_language.Layer):
    m = SLD(layer.material.get_sld() * 1e6)
    return Slab(
        layer.thickness.as_unit("angstrom"),
        m,
        layer.roughness.as_unit("angstrom"),
        name=getattr(layer, "original_name", ""),
    )


def res_leaflet(leaflet: model_complex.Leaflet):
    apm = leaflet.apm.as_unit("angstrom^2")
    vm_heads = leaflet.heads.volume.as_unit("angstrom^3")
    vfrac_head = 1.0 - leaflet.heads_hydration
    vfrac_tail = 1.0 - leaflet.tails_hydration
    b_heads = leaflet.heads.get_sld() * 1e6 / vm_heads
    d_heads = vm_heads / apm / vfrac_head
    vm_tails = leaflet.tails.volume.as_unit("angstrom^3")
    b_tails = leaflet.tails.get_sld() * 1e6 / vm_tails
    d_tails = vm_tails / apm / vfrac_tail
    solvent = leaflet.solvent.get_sld() * 1e6
    ll = LipidLeaflet(
        apm=apm,
        b_heads=b_heads,
        vm_heads=vm_heads,
        thickness_heads=d_heads,
        b_tails=b_tails,
        vm_tails=vm_tails,
        thickness_tails=d_tails,
        rough_head_tail=leaflet.roughness.as_unit("angstrom"),
        rough_preceding_mono=leaflet.roughness.as_unit("angstrom"),
        reverse_monolayer=not leaflet.heads_first,
        name="LL " + getattr(leaflet, "original_name", ""),
        head_solvent=solvent,
        tail_solvent=solvent,
    )
    return ll


def res_bilayer(bilayer: model_complex.Bilayer):
    apm = bilayer.apm.as_unit("angstrom^2")
    vm_heads = bilayer.outer.volume.as_unit("angstrom^3")
    vfrac_head = 1.0 - bilayer.outer_hydration
    vfrac_tail = 1.0 - bilayer.inner_hydration
    b_heads = bilayer.outer.get_sld() * 1e6 / vm_heads
    d_heads = vm_heads / apm / vfrac_head
    vm_tails = bilayer.inner.volume.as_unit("angstrom^3")
    b_tails = bilayer.inner.get_sld() * 1e6 / vm_tails
    d_tails = vm_tails / apm / vfrac_tail
    solvent = bilayer.solvent.get_sld() * 1e6
    ll = LipidLeaflet(
        apm=apm,
        b_heads=b_heads,
        vm_heads=vm_heads,
        thickness_heads=d_heads,
        b_tails=b_tails,
        vm_tails=vm_tails,
        thickness_tails=d_tails,
        rough_head_tail=bilayer.roughness.as_unit("angstrom"),
        rough_preceding_mono=bilayer.roughness.as_unit("angstrom"),
        reverse_monolayer=False,
        name="LL " + getattr(bilayer, "original_name", ""),
        head_solvent=solvent,
        tail_solvent=solvent,
    )
    vfrac_head = 1.0 - (bilayer.outer_hydration_2 or bilayer.outer_hydration)
    vfrac_tail = 1.0 - (bilayer.inner_hydration_2 or bilayer.inner_hydration)
    d_heads = vm_heads / apm / vfrac_head
    d_tails = vm_tails / apm / vfrac_tail
    rll = LipidLeaflet(
        apm=apm,
        b_heads=b_heads,
        vm_heads=vm_heads,
        thickness_heads=d_heads,
        b_tails=b_tails,
        vm_tails=vm_tails,
        thickness_tails=d_tails,
        rough_head_tail=bilayer.roughness.as_unit("angstrom"),
        rough_preceding_mono=bilayer.roughness.as_unit("angstrom"),
        reverse_monolayer=True,
        name="rLL " + getattr(bilayer, "original_name", ""),
        head_solvent=solvent,
        tail_solvent=solvent,
    )
    return ll | rll


refnx_resolvers = {
    model_language.Layer: res_layer,
    model_complex.Leaflet: res_leaflet,
    model_complex.Bilayer: res_bilayer,
}


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

    layers = sample.resolve_to_layers()
    print("\n".join([repr(li) for li in layers]))

    structure = Structure()
    for block in blocks:
        if type(block) in refnx_resolvers:
            structure |= refnx_resolvers[type(block)](block)
        else:
            for li in block.resolve_to_layers():
                structure |= res_layer(li)
    print("\n", structure, "\n")
    model = ReflectModel(structure, bkg=0.0)
    structurex = Structure()
    for lj in layers:
        m = SLD(lj.material.get_sld(xray_energy="Cu") * 1e6)
        structurex |= m(lj.thickness.as_unit("angstrom"), lj.roughness.as_unit("angstrom"))
    modelx = ReflectModel(structurex, bkg=0.0)

    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(121)
    pyplot.semilogy(q, model(q), label="neutron")
    pyplot.semilogy(q, modelx(q), label="x-ray (Cu)")
    pyplot.legend()
    pyplot.title(txt)
    pyplot.xlabel("q [Å$^{-1}$]")
    pyplot.ylabel("Neutron-reflectivity")
    pyplot.subplot(122)
    pyplot.plot(*structure.sld_profile(), label="neutron")
    pyplot.plot(*structurex.sld_profile(), label="x-ray (Cu)")
    pyplot.legend()
    pyplot.title(txt)
    pyplot.ylabel("SLD / $10^{-6} \\AA^{-2}$")
    pyplot.xlabel("distance / $\\AA$")
    pyplot.show()


if __name__ == "__main__":
    main()
