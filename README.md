# gpSTS (powered by gpCAM) tutorial library
A Python interface to run autonomous bias spectroscopy measurements.

## Requirements & Overview

Python 3.8 or higher

from CLI:

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

## Citing

If you use `gpSTS` in your work, please cite the accompanying paper:

```bibtex
@incollection{noack2023methods,
author={Rossi, Antonio and Smalley, Darian and Ishigami, Masahiro and Rotenberg, Eli and Weber-Bargioni, Alexander and Thomas, John C.},
title={Autonomous Hyperspectral Scanning Tunneling Spectroscopy within Methods and Applications of Autonomous Experimentation},
editor = "Noack, Marcus and Ushizima, Daniela",
publisher = "CRC Press",
booktitle = "Methods and Applications of Autonomous Experimentation",
year={2023}
}