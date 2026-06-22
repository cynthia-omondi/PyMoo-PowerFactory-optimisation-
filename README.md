# PyMoo-PowerFactory-optimisation
Multiobjective grid optimisation using PyMoo and PowerFactory

=================================================
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17961683.svg)](https://doi.org/10.5281/zenodo.17961683)

## Overview

This repository contains a customized implementation of the PyMOO optimization framework developed to support the methodology presented in this research. Several components of the original PyMOO codebase were modified to accommodate problem-specific requirements and algorithmic enhancements.

The repository includes the optimization notebook used to execute experiments as well as the modified source files required for reproducing the results.

---

## Repository Structure

### MY FUNC.ipynb

Primary notebook for running the optimization workflow.

Functions include:
- Problem setup and configuration
- Decision variable definition
- Objective function evaluation
- Constraint handling
- Execution of the optimization algorithm
- Post-processing and visualization of results

This notebook serves as the main entry point for reproducing the optimization studies.

---

### pymoo/core.py

Modified PyMOO source file.

Changes were introduced to support:
- Custom optimization logic
- Problem-specific constraint handling
- Enhanced solution evaluation procedures
- Additional functionality required by the proposed methodology

Refer to the code comments for detailed descriptions of individual modifications.

---

## Citation


## Contact
For more information regarding the methodology, its functionality and implementation
please contact the author at dorcus70@gmail.com
