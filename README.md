# FairBench

![build](https://github.com/mever-team/FairBench/actions/workflows/tests.yml/badge.svg)
![coverage](coverage.svg)
[![Documentation Status](https://readthedocs.org/projects/fairbench/badge/?version=latest)](https://fairbench.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

A comprehensive AI fairness exploration framework.
- :chart_with_upwards_trend: Fairness reports and stamps
- :flags: Multivalue multiattribute
- :hammer_and_wrench: Measure building blocks
- :gear: ML integration (`numpy`,`torch`,`tensorflow`,`jax`)

*FairBench strives to be compatible with the latest Python release, 
but compatibility delays of third-party ML libraries usually 
means that only the language's previous release is tested and
stable (currently 3.12).*

## [Documentation](https://fairbench.readthedocs.io/)

## [Benchmarks](benchmarks/README.md)

## Attributions

```
@article{krasanakis2024standardizing,
      title={Towards Standardizing AI Bias Exploration}, 
      author={Emmanouil Krasanakis and Symeon Papadopoulos},
      year={2024},
      eprint={2405.19022},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**Maintainer:** Emmanouil (Manios) Krasanakis (maniospas@hotmail.com)<br>
**License:** Apache 2.0<br>
**Contributors:** Giannis Sarridis


This project includes modified code originally licensed under the MIT License:

- ReBias. (Copyright © 2020-present NAVER Corp)<br>
Modifications © 2024 Emmanouil Krasanakis.<br>
See [fairbench/bench/vision/datasets/mnist/](fairbench/bench/vision/datasets/mnist/) for details.
