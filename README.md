This is a development version of cosmotracer, a package used to understand landscapes
by combining the powers of cosmogenic nuclides and geomorphic analyses.

It is based on and inherits many functionalities of [landlab](https://github.com/landlab/landlab), the versatile Python-based tool for
modelling and understanding all sorts of processes in landscapes.

This code is in the early stages of development, but some aspects might already be interesting:
1. `CosmoLEM` is a wrapper around `landlab`'s `RasterModelGrid` and can be used to model the TCN
signal of an evolving landscape. 
2. `Basin`, which inherits the capabilities of `CosmoLEM`, is a class used to
extract and handle drainage basins from DEMs.
3. `cosmotracer.tcn.calculate_transient_concentrations` can be used to calculate
TCN concentrations for non-steady-state exhumation histories.

More will be added in the future.

Developed by Lennart Grimm, UCL