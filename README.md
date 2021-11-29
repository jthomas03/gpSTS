# gpSTS (powered by gpCAM) library within Nanonis/LabVIEW framework
A Python-LabVIEW interface to run autonomous bias spectroscopy measurements.

## Requirements

Nanonis v4.5 (SPECS) controlled SPM setup.
The nanonis_programming_interface_v4.5 or newer needs to be installed.
LabView (2019 or higher) needs to be installed.
Python 3.8 or higher

from CLI with Nanonis controller, labVIEW application, and bias spectroscopy module
running:

```
~python Run_gpSTS.py
```

after a completed autonomous experiment, from CLI:

```
~python Run_postprocessing.py
```

after postprocessing is completed, training or classification with a trained model can be performed, from CLI:

```
~python Run_CNN.py
```

Experimental configuration and hyperparameter specification is in: 

```
Config.py
```

## Model Training

The structure of the training path is shown below, where separate ```train``` and ```validation``` folders are used for initial loading.

    ├── train
    │   ├── class1
    │   ├── class2
    │   ├── class3
    │   ├── class4
    └── validation
        ├── class1
        ├── class2
        ├── class3
        ├── class4

**Dataset used in publication is provided at:** http://doi.org/10.5281/zenodo.4633866.

## Python Dependencies

The following dependencies are required, and should be available from PyPi.

* ```numpy```   — support for large, multi-dimensional arrays
* ```matplotlib``` — visualization tool
* ```scipy``` — scientific and technical computing library
* ```gpcam``` — library for autonomous experimentation
* ```fvgp``` — library for highly flexible function-valued Gaussian processes
* ```torch``` — library for tensor computation and deep neural networks
* ```torchvision``` — library for computer vision

## Citing

If you use `gpSTS` in your work, please cite the accompanying paper:

```bibtex
@article{thomas2021autonomous,
      title={Autonomous Scanning Probe Microscopy Investigations over WS₂ and Au{111}}, 
      author={John C. Thomas and Antonio Rossi and Darian Smalley and Luca Francaviglia and Zhuohang Yu and Tianyi Zhang and Shalini Kumari and Joshua A. Robinson and Mauricio Terrones and Masahiro Ishigami and Eli Rotenberg and Edward Barnard and Archana Raja and Ed Wong and D. Frank Ogletree and Marcus Noack and Alexander Weber-Bargioni},
      year={2021},
      eprint={2110.03351},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2110.03351}
}