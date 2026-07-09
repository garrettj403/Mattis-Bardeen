from setuptools import setup

setup(
    name="Mattis-Bardeen",
    version="0.0.1dev",
    author="John Garrett",
    author_email="garrettj403@gmail.com",
    description=("Calculate the electrical properties of superconductors using Mattis-Bardeen theory."),
    license="GPL v3",
    keywords="superconductivity, physics, superconducting detectors, terahertz instrumentation, Python",
    py_modules=["mattisbardeen"],
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'scipy'
    ],
    long_description="Calculate the electrical properties of superconductors using Mattis-Bardeen theory.",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
