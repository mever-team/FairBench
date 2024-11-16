import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

# with open("README.md", "r") as file:
#    long_description = file.read()

long_description = (
    "A comprehensive AI fairness exploration framework.<br>"
    "**Homepage:** https://fairbench.readthedocs.io<br>"
    "**Repository:** https://github.com/mever-team/FairBench"
)

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

with open("requirements[interactive].txt", "r") as file:
    interactive_requirements = file.read().splitlines()

setuptools.setup(
    name="fairbench",
    version="0.3.16",
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
    install_requires=requirements,
    extras_require={"interactive": interactive_requirements},
)
