from setuptools import setup, find_packages

setup(
    name="Mattis-Bardeen",
    version="0.0.1dev",
    author="John Garrett",
    author_email="garrettj403@gmail.com",
    description=("Calculate the electrical properties of superconductors using Mattis-Bardeen theory."),
    license="GPL v3",
    keywords="superconductivity, physics, superconducting detectors, terahertz instrumentation, Python",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    long_description="Calculate the electrical properties of superconductors using Mattis-Bardeen theory.",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
