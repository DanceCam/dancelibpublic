# DanceCam
The public repository for **DanceCam**, showcasing some of the features of our atmospheric turbulence mitigation method.

The **project website** is available at https://dancecam.info.

The **paper** is available [here](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stae1018/7654005).


### Requirements (will be installed automatically by following the installation instructions)

- Python 3.8+
- PyTorch 1.10.0+
- NumPy 1.21.2+
- Astropy 4.0.1+
- tqdm 4.62.3+
- Blended Tiling 0.1.0+
- SEWpy 0.1.0+

### Installation

1. Clone the repository:

```bash
git clone https://github.com/DanceCam/dancelibpublic.git
```

2. Install:

```bash
cd dancelibpublic
pip install -e .
```

### Usage

The *demo* folder contains a simple Jupyter notebook that shows how to use the library to perform inference on a video stream.

### Citing DanceCam

To cite DanceCam, please use the following BibTeX entry:

```
@article{bialek2024dancecam,
  title = {DanceCam: atmospheric turbulence mitigation in wide-field astronomical images with short-exposure video streams},
  author = {Bialek, Spencer and Bertin, Emmanuel and Fabbro, S{\'e}bastien and Bouy, Herv{\'e} and Rivet, Jean-Pierre and Lai, Olivier and Cuillandre, Jean-Charles},
  journal = {Monthly Notices of the Royal Astronomical Society},
  pages = {stae1018},
  year = {2024},
  url = {https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stae1018/7654005},
  publisher = {Oxford University Press},
}
```