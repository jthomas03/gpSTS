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

## Python Dependencies

The following dependencies are required, and should be available from PyPi.

* ```numpy```   — support for large, multi-dimensional arrays
* ```matplotlib``` — visualization tool
* ```scipy``` — scientific and technical computing library
* ```gpcam``` — library for autonomous experimentation
* ```pytorch``` — library for tensor computation and deep neural networks
