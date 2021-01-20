# dependencies
* python             for ML
* python:pytorch     for learning contemporary ML models
* python:pillow      for reading images
* python:scipy       for UL classifier
* python:opencv      for low level image manipulation
* python:pygame      for visualization

* povray             for rendering synthetic test and training scenes

## install
conda is recommended for managing the python environment

    conda create --name ml
    conda activate ml
    conda install -n ml pytorch
    conda install -n ml pillow
    conda install -n ml scipy
    conda install -c conda-forge opencv
    
    conda activate ml
    python3 -m pip install -U pygame --user

#models
This system needs (pretrained) models. Good updated models will be linked here.

[01/20/2021](https://drive.google.com/file/d/16fGJp-2tY2CO2GkRyGkie2tDQQXpAftS/view?usp=sharing) trained for all existing motions, reaches a testset accuracy of ~0.88<br/>
