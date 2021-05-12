Installation
============

The Displacement-strain-planet package requires `pyshtools <https://github.com/SHTOOLS/SHTOOLS/>`_ (>=4.7.1). This package can be installed using either `conda` or `pip` ::

    conda install -c conda-forge pyshtools  # Linux and macOS only
    pip install pyshtools

After `pyshtools` is installed, you can install the Displacement-strain-planet module using the command ::

    pip install Displacement-strain-planet

**Working with the example scripts**

To access the example scripts, you must download the entire Displacement_strain_planet
repository from GitHub. The easiest way to do this is by cloning the repo::

    git clone https://github.com/AB-Ares/Displacement_strain_planet.git
    cd Displacement_strain_planet/

If Displacement-strain-planet was not installed using the `pip` command above, it can be installed from the downloaded source using one of the two commands::

    pip install .

will install Displacement-strain-planet in the active Python environment lib folder, whereas ::

    pip install -e .

will install the files in the current working directory and link them to the system Python directory. The second method is preferred if you plan on modifying the Displacement-strain-planet source code.

To execute a script, it is only necessary to enter the `examples` directory and to run the file using the python command ::

    cd examples
    python Mars_crust_displacement.py 

You can also run the detailed jupyter notebook with ::

    cd examples
    jupyter notebook Run_demo.ipynb 

If jupyter notebook is not installed, you can install it with ::

    pip install notebook

.. note::
    Depending on how your system is set up, it might be necessary to use
    explicitly ``python3`` and ``pip3`` instead of ``python`` and ``pip`` in
    the above commands.
