Examples
========

.. note::
    In order to access these example files, it will be necessary to either download or clone the entire Displacement-strain-planet repo from `GitHub <https://github.com/AB-Ares/Displacement_strain_planet>`_. The files will be located in the directory `examples`. To run the jupyter notebook, you must install jupyter notebooks with pip (see Installation).

Mars
----

``Mars_crust_displacement.py``
    A script that demonstrates how to calculate the moho-relief on Mars using global gravity and topography data. The moho relief is splited in an isostatic part and a displacement part, which depends on the elastic thickness of the lithosphere. The script then computes the principal horizontal strains and their directions associated with the displacement.

``Run_demo.ipynb``
    A jupyter notebook that shows many of the functionalities of Displacement-strain-planet: moho-relief calculations under various assumptions, including Airy or Pratt isostasy, displacement calculations due to a mantle plume underneath Tharsis or due to internal loading in phase with the surface topography, strain calculations. 