"""Recover a known discrete state potential with frozen-bias production."""

import copy

import numpy as np

import metatally as mt

rng = np.random.default_rng(4)
n_variables = 2
radix = 3
space0 = mt.TorsionChain(n_variables=n_variables, radix=radix).state_space()
uxi = rng.normal(scale=0.8, size=space0.n_states)
uxi -= uxi[0]

model = mt.TorsionChain(
    n_variables=n_variables,
    radix=radix,
    barrier=1.5,
    state_potential=uxi,
)
space = model.state_space()

#bias = mt.CompositeBias([
#    mt.DiscreteStateBias.from_space(space, height=0.05),
#    mt.IndependentTorsionBias.from_space(space, height=0.01, sigma=0.2, n_grid=96),
#])

bias = mt.CompositeBias([
    mt.WellTemperedDiscreteStateBias.from_space(
        space,
        height=0.2,
        bias_factor=10.0,
        beta=1.0,
    ),
    mt.IndependentTorsionBias.from_space(
        space,
        height=0.05,
        sigma=0.2,
        n_grid=96,
    ),
])


adaptive = mt.MetropolisSampler(
    model=model,
    state_space=space,
    bias=bias,
    initial=model.initial_state(),
    step_size=np.pi / 10,
    seed=1,
).run(50_000)

frozen_bias = copy.deepcopy(adaptive.bias)
frozen_bias.freeze()

production = mt.MetropolisSampler(
    model=model,
    state_space=space,
    bias=frozen_bias,
    initial=adaptive.x,
    step_size=np.pi / 10,
    seed=2,
).run(50_000)

estimated = mt.reweighted_free_energy_differences(
    production.states,
    production.bias_values,
    n_states=space.n_states,
)

print("xi  true_dF  estimated_dF")
for xi, (true, est) in enumerate(zip(model.state_free_energy_differences(), estimated)):
    print(f"{xi:2d}  {true:8.3f}  {est:12.3f}")

print(mt.reweighting_diagnostics(production.bias_values))
