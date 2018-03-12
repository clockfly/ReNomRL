from setuptools import find_packages, setup

requires = [
    "numpy", "pandas", "scikit-learn", "scipy", "future", "renom"
]

setup(
    install_requires=requires,
    name="renom_rl",
    version="0.0beta",
    packages=find_packages(),
)