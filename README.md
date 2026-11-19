# IFCA

Python program for Independent Fluorescence Component Analysis.

## Usage

Copy `IFCA.py` module to your working directory.

### From Python scripts or Jupyter notebooks
Call functions in `IFCA.py` from your Python script as exemplified in the following tutorial notebooks.

* `Analysis on bh photon data.ipynb`: for photon data files in the Becker & Hickl's `.spc` format.
* `Analysis on pq photon data.ipynb`: for photon data files in the PicoQuant's `.ptu` format.

### Use of GUI scripts

Copy `IFCA_GUI_xx.py` to the same directory and run it. `xx` is either `bh` (Becker & Hickl) or `pq` (PicoQuant), depending on the format of photon data.

## Requirement

`IFCA.py` runs under Python and requires Numpy, SciPy, and Numba packages. Their versions used for testing are as follows.

* Python 3.13.1
* NumPy 2.1.3
* SciPy 1.15.1
* Numba 0.61.0

Tutorial notebooks are in Jupyter Notebook format (`.ipynb`) and require a Jupyter environment (such as Google Colab) to read.

For loading experimentally obtained photon data, it is convenient to use [*phconvert*](https://photon-hdf5.github.io/phconvert)[1]. (Note: Currently, our codes are not compatible with `phconvert 0.10.x`. Please use `phconvert 0.9.1`.) The GUI scripts requires `phconvert` and `matplotlib`.

[1] A. Ingargiola et al., Biophys. J. 110, 26-33 (2016).

## Author

This program is developed by Kunihiko Ishii and Miyuki Sakaguchi from Molecular Spectroscopy Laboratory, RIKEN.  
Contact: kishii@a.riken.jp (Kunihiko Ishii)

## License

Copyright &copy; 2025 RIKEN. This program is licensed under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).