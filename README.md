# Kolmogorov-Arnold Mixture of Experts

 
Kolmogorov-Arnold Mixture of Experts is a Mixture of Expert framework that can be applied on any Layer or Model. 
This Keras implementation integrates KAMoE (and MoE) as a layer that takes as inputs in initialization another layer or model, and create a mixture of experts framework over it. 
It is compatible with any backend, and can be used with any layer or model.
In case of sequential model, the MoE (or KAMoE) will look if the input shape has the same number of dimension in input and output, if there is one less dimension it will search for a layer with a return_sequences=False in the expert. In that case it will only use the last element of the sequence for the gating, otherwise it will flatten the whole sequence to use it as input for the gating. In any other case it will use the whole input as input for the gating.
The implementation is tested to be compatatible with Tensorflow, Jax and Torch. From testing jax is the best backend in terms of performance with it, while torch is very slow (mostly due to keras handling of pytorch than anything else for the moment I believe).
It is the original implementation of the [paper](https://arxiv.org/abs/2409.15161)
The KAN part implementation has been inspired from [efficient_kan](https://github.com/Blealtan/efficient-kan), and is available [here](https://github.com/remigenet/keras_efficient_kan) and works similarly to it, thus not exactly like the [original implementation](https://github.com/KindXiaoming/pykan).

In case of performance consideration, the best setup tested used [nvidia cuda docker image](https://hub.docker.com/r/nvidia/cuda) followed by installing jax using ```pip install "jax[cuda12]"```, this is what is used in the example section.
I also discourage using as is the example for torch, it seems that currently when running test using torch backend with keras is much slower than torch directly, even for GRU or LSTM. 

![KAMoE representation]()

## Installation

Install KAMoE directly from PyPI:

```bash
pip install kamoe
```

Dependencies are managed using pyproject.toml.

## Usage

TKAN can be used within keras model easily.
Here is an example that demonstrates how to use KAMoE in a sequential model:

```python
import keras
from kamoe import MoE, KAMoE
from keras.models import Sequential

model = Sequential([
    Input(shape=(10,)),
    KAMoE(Dense(100, activation='relu'), n_experts=10, gating_activation='softmax'),
    Dense(100, activation='relu'),
    Dense(units=n_ahead, activation='linear')
], name = model_id)
```

You can also use it in a functional model for example doing:

```python
def create_nn_model(model_type, variant, n_layers, n_units, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for _ in range(n_layers):
        if model_type == 'MLP':
            if variant == 'Standard':
                model.add(Dense(n_units, activation='relu'))
            elif variant == 'MoE':
                model.add(MoE(Dense(n_units, activation='relu')))
            elif variant == 'KAMoE':
                model.add(KAMoE(Dense(n_units, activation='relu')))
        elif model_type == 'KANLinear':
            if variant == 'Standard':
                model.add(KANLinear(n_units))
            elif variant == 'MoE':
                model.add(MoE(KANLinear(n_units)))
            elif variant == 'KAMoE':
                model.add(KAMoE(KANLinear(n_units)))
    
    model.add(Dense(1, activation='linear'))
    return model
```

Finally you can also use it in a more complex model, for example doing:

```python
base_model = Sequential([
    Input(shape=(10,)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu')
])
model = Sequential([
    Input(shape=(10,)),
    KAMoE(base_model, n_experts=10, gating_activation='softmax'),
    Dense(100, activation='relu'),
    Dense(units=n_ahead, activation='linear')
], name = model_id)
```

You can find a more complete example with comparison with other models in the example folder.

Please cite our work if you use this repo:

```
@article{genet2024kamoe,
  title={A Gated Residual Kolmogorov-Arnold Networks for Mixtures of Experts},
  author={Inzirillo, Hugo and Genet, Remi},
  journal={arXiv preprint arXiv:2409.15161},
  year={2024}
}
```

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
