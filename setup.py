#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import versioneer

versioneer.versionfile_source = 'Displacement_strain_planet/_version.py'
versioneer.versionfile_build = 'Displacement_strain_planet/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'Displacement_strain_planet-'

# Convert markdown README.md to restructured text (.rst) for PyPi
try:
    import pypandoc
    rst = pypandoc.convert_file('README.md', 'rst')
    long_description = rst.split('\n', 5)[5]
except(IOError, ImportError):
    print('*** pypandoc is not installed. PYPI description will not be '
          'formatted correctly. ***')
    long_description = open('README.md').read()

install_requires = ['pyshtools>=4.7.1', 'sympy<=1.7']

setup(name='Displacement_strain_planet',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Planetary crustal thickness, displacement, ' +
      'stress and strain calculations in spherical harmonics.',
      long_description=long_description,
      url='https://github.com/AB-Ares/Displacement_strain_planet',
      author='Adrien Broquet',
      author_email='adrien.broquet@oca.eu',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering'
      ],
      keywords=['stress', 'strain', 'flexure', 'elastic thickness',
                'crust', 'gravity', 'geophysics'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires,
      python_requires='>=3.7')
