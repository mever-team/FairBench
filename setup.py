import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

#with open("README.md", "r") as file:
#    long_description = file.read()

long_description = "For tutorials, documentation, and contribution guidelines, please visit the project's homepage at https://github.com/mever-team/FairBench"

setuptools.setup(
    name='fairbench',
    version='0.2.1',
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Fairness model assessment framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mever-team/FairBench",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
    install_requires=[
        'numpy', 'eagerpy', 'distributed', 'makefun', 'matplotlib', 'wget', 'scikit-learn', 'pandas', 'objwrap'
    ],
 )