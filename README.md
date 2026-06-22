# Multiobjective grid optimisation using PyMoo and PowerFactory
[![DOI](https://zenodo.org/badge/1277269834.svg)](https://doi.org/10.5281/zenodo.20802531)

## Overview

This repository contains code for the implementation of multi-objective grid optimisation linking PyMoo and PowerFactory.
It includes the optimization notebook and the modified source files.

## Repository Structure

- ### README.rst
PyMoo installation details and contact information and citation information for source code.

- ### MY FUNC.ipynb
Primary notebook for running the optimization workflow. Functions include: Problem setup and configuration, Decision variable definition, Objective function evaluation, Constraint handling, and execution of the optimization algorithm

- ### pymoo/core/mixed.py
Modified PyMOO source file. Changes were introduced to support custom crossover and mutation functions

---

## Citation (for new functions and modifications only)
Omondi, C. (2026). Multiobjective grid optimisation using PyMoo and PowerFactory (Version v1). Zenodo. https://doi.org/10.5281/zenodo.20802532

Note: For citing source code please refer to README.rst

## Contact
For more information regarding the modifications, functionality and implementation
please contact the author at dorcus70@gmail.com
