import copy

import numpy as np

import metatally as mt


def test_sampler_runs_and_can_continue():
    model = mt.TorsionChain(n_variables=2, radix=3, barrier=2.0)
    space = model.state_space()
    sampler = mt.MetropolisSampler(model=model, state_space=space, seed=1, step_size=0.5)
    sampler.run(10)
    assert sampler.n_steps == 10
    sampler.run(5)
    assert sampler.n_steps == 15
    assert sampler.states.shape == (15,)
    assert sampler.bias_values.shape == (15,)


def test_sampler_frozen_bias_does_not_change():
    model = mt.TorsionChain(n_variables=1, radix=3, barrier=1.0)
    space = model.state_space()
    bias = mt.DiscreteStateBias.from_space(space, height=1.0)
    sampler = mt.MetropolisSampler(model=model, state_space=space, bias=bias, seed=2)
    sampler.run(10)
    frozen = copy.deepcopy(bias)
    frozen.freeze()
    before = frozen.array.copy()
    prod = mt.MetropolisSampler(model=model, state_space=space, bias=frozen, initial=sampler.x, seed=3)
    prod.run(10)
    np.testing.assert_allclose(frozen.array, before)
