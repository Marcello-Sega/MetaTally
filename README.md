# MetaTally

State-counting tools for sampling combinatorial metastable spaces.

## Core idea

MetaTally maps local state labels to a single global state index:

```python
import metatally as mt

enc = mt.ternary_encoder(3)
xi = enc.encode([2, 0, 1])
assert xi == 11
assert enc.decode(xi).tolist() == [2, 0, 1]
```

Use `StateSpace` when raw coordinates first need to be assigned to local states:

```python
import numpy as np
import metatally as mt

assigner = mt.TorsionStateAssigner(
    centers=[-2*np.pi/3, 0.0, 2*np.pi/3],
    n_variables=3,
)
space = mt.StateSpace(assigner)

values = [0.1, 2.0, -2.0]
print(space.assign(values).tolist())  # [1, 2, 0]
print(space.encode(values))           # 7
```

## Torsion-chain model

`TorsionChain` is a small test model for sampling methods:

```python
model = mt.TorsionChain(
    n_variables=6,
    radix=3,
    barrier=4.0,
    coupling=[0.0, 0.5, 1.0, 0.5, 0.0],
)
space = model.state_space()
```

The energy contains local torsional barriers and optional nearest-neighbour
couplings,

```text
U(phi) = k sum_i [1 - cos(q phi_i)] - sum_i J_i cos(phi_{i+1} - phi_i)
```

and can also include a known discrete state potential:

```python
uxi = np.linspace(0.0, 2.0, space.n_states)
model = mt.TorsionChain(n_variables=2, radix=3, barrier=2.0, state_potential=uxi)
```

This is useful for checking whether a sampling/reweighting workflow can recover
known state free-energy differences.

## Biases

The current bias objects include:

```python
mt.DiscreteStateBias          # V_xi(xi)
mt.IndependentTorsionBias     # sum_n V_n(phi_n)
mt.StateConditionedTorsionBias # sum_n W_n(xi, phi_n)
mt.CompositeBias              # sum of several bias objects
```

Example combined bias:

```python
bias = mt.CompositeBias([
    mt.DiscreteStateBias.from_space(space, height=0.05),
    mt.IndependentTorsionBias.from_space(space, height=0.01, sigma=0.2),
])
```

## Stateful sampling

```python
sampler = mt.MetropolisSampler(
    model=model,
    state_space=space,
    bias=bias,
    initial=model.initial_state(),
    step_size=np.pi/12,
    beta=1.0,
    seed=1,
)

for _ in range(10000):
    sampler.step()

print(sampler.coverage)
print(sampler.n_unique_by_step)
```

## Frozen-bias production

Adaptive MetaD-like trajectories are not usually suitable for direct static
reweighting. A simple workflow is to build a bias, freeze it, then run a fixed-
bias production trajectory:

```python
import copy

adaptive = sampler.run(20000)
frozen_bias = copy.deepcopy(adaptive.bias)
frozen_bias.freeze()

production = mt.MetropolisSampler(
    model=model,
    state_space=space,
    bias=frozen_bias,
    initial=adaptive.x,
    step_size=np.pi/12,
    beta=adaptive.beta,
    seed=2,
).run(100000)

probs = mt.reweighted_state_probabilities(
    production.states,
    production.bias_values,
    beta=production.beta,
    n_states=production.state_space.n_states,
)

dF = mt.reweighted_free_energy_differences(
    production.states,
    production.bias_values,
    beta=production.beta,
    n_states=production.state_space.n_states,
    reference=0,
)
```

## Utilities

Scan one coordinate while keeping the rest fixed:

```python
phi, U = mt.scan_coordinate(model, variable=0, background=model.initial_state())
```

Run tests:

```bash
pytest
```
