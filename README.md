# DeepSolid

An implementation of the algorithm given in 
["Ab initio calculation of real solids via neural network ansatz"](https://rdcu.be/c4rNI). 
A periodic neural network is proposed as wavefunction ansatz for solid quantum Monte Carlo and achieves 
unprecedented accuracy compared with other state-of-the-art methods.
This repository is developed upon [FermiNet](https://github.com/deepmind/ferminet/tree/jax) 
and [PyQMC](https://github.com/WagnerGroup/pyqmc). 

## Installation

DeepSolid can be installed via the supplied setup.py file.
```shell
# Install with CPU only
pip3 install -e . -f https://storage.googleapis.com/jax-releases/jax_releases.html
# or with GPU
pip3 install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Python 3.9 is recommended.
If GPU is available, we recommend you to install jax and jaxlib with cuda 11.4+. 
Our experiments were carried out with `jax==0.2.26` and `jaxlib==0.1.75.` 

## Usage

[Ml_collection](https://github.com/google/ml_collections) package is used for system definition. Below is a simple example of H10 in PBC:
```
deepsolid --config=PATH/TO/DeepSolid/config/two_hydrogen_cell.py:H,5,1,1,2.0,0,ccpvdz --config.batch_size 4096
```

### Customize your system
Simulation system can be customized in config.py file, such as

```python
import numpy as np
from pyscf.pbc import gto
from DeepSolid import base_config
from DeepSolid import supercell


def get_config(input_str):
    symbol, S = input_str.split(',')
    cfg = base_config.default()

    # Set up cell.
    cell = gto.Cell()
    
    # Define the atoms in the primitive cell.
    cell.atom = f"""
    {symbol} 0.000000000000   0.000000000000   0.000000000000
    """
    
    # Define the pretrain basis.
    cell.basis = "ccpvdz"
    
    # Define the lattice vectors of the primitive cell.
    # In this example it's a simple cubic.
    cell.a = np.array([[3.0, 0.0, 0.0],
                       [0.0, 3.0, 0.0],
                       [0.0, 0.0, 3.0]])
    
    # Define the unit used in cell definition, only support Bohr now. 
    cell.unit = "B"
    cell.verbose = 5
    
    # Define the threshold to discard gaussian basis used in pretrain.
    cell.exp_to_discard = 0.1
    cell.build()
    
    # Define the supercell for QMC, S specifies how to tile the primitive cell.
    S = np.eye(3) * int(S)
    simulation_cell = supercell.get_supercell(cell, S)
    
    # Assign the defined supercell to cfg.
    cfg.system.pyscf_cell = simulation_cell

    return cfg
```
After defining the config file, simply use the following command to launch the simulation:

```shell
deepsolid --config=PATH/TO/config.py:He,1 --config.batch_size 4096
```


### Read structure from poscar file

We also support reading structure from poscar file, which is commonly used. Simply use the following command
```shell
deepsolid --config=DeepSolid/config/read_poscar.py:PATH/TO/POSCAR/bcc_li.vasp,1,ccpvdz
```
## Distributed training
Present released code doesn't support multi-node training. See [this link](https://github.com/google/jax/pull/8364)
for help.

## Tricks to accelerate
The bottleneck of DeepSolid is the laplacian evaluation of the neural network. We recommend 
the users to use partition mode instead, simply adding two more flags:
```shell
deepsolid --config=PATH/TO/config.py --config.optim.laplacian_mode=partition --config.optim.partition_number=3
```
Partition mode will try to parallelize the calculation of laplacian and partition number must be a factor of 
(electron number * 3). Note that partition mode will require a lot of GPU memory.

## Precision
DeepSolid supports both FP32 and FP64. However, we recommend the users turn off the TF32 mode which 
is automatically adopted in A100 if FP32 is chosen. TF32 can be turned off using the following command:

```shell
NVIDIA_TF32_OVERRIDE=0 deepsolid --config.use_x64=False
```

## Giving Credit

If you use this code in your work, please cite the associated paper.

```
@article{li2022ab,
  title={Ab initio calculation of real solids via neural network ansatz},
  author={Li, Xiang and Li, Zhe and Chen, Ji},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={7895},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
