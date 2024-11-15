from setuptools import setup, find_packages

setup(
    name="combss",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",       # Numerical computations
        "scipy>=1.7.0",        # Sparse matrix operations
        "scikit-learn>=1.0.0"  # Machine learning tools and metrics
    ],
    author="Sarat Moka and Hua Yang Hu",
    description="A package implementation of COMBSS, a novel continuous optimisation method toward best subset selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/saratmoka/combss",
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)