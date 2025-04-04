import setuptools

# Developer self-reminder for uploading to PyPI:
# - Install: wheel, twine
# - Build  : python setup.py sdist bdist_wheel
# - Deploy : twine upload dist/*
#
# Coverage for v2: pytest tests/experimental --cov=fairbench/experimental --cov-report=html

long_description = (
    "A comprehensive AI fairness exploration framework.<br>"
    "**Homepage:** https://fairbench.readthedocs.io<br>"
    "**Repository:** https://github.com/mever-team/FairBench"
)


def read_requirements(filename):
    try:
        with open(filename, "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using an empty list of requirements.")
        return []


requirements = read_requirements("requirements.txt")
interactive_requirements = read_requirements("requirements[interactive].txt")
vision_requirements = read_requirements("requirements[vision].txt")
llm_requirements = read_requirements("requirements[llm].txt")
graph_requirements = read_requirements("requirements[graph].txt")

# Setup configuration
setuptools.setup(
    name="fairbench",
    version="0.7.17",
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="A fairness assessment framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://fairbench.readthedocs.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "interactive": interactive_requirements,
        "vision": vision_requirements,
        "graph": graph_requirements,
        "llm": llm_requirements,
    },
)
