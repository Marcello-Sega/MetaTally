"""Compare plain and biased MC on a torsion-chain toy model."""

from __future__ import annotations

import matplotlib.pyplot as plt

import metatally as mt


def main() -> None:
    """Run the comparison and show a discovery plot."""
    model = mt.TorsionChain(n_variables=6, radix=3, barrier=4.0)
    space = model.state_space()
    x0 = model.initial_state()
    n_steps = 10_000

    plain = mt.MetropolisSampler(
        model,
        space,
        initial=x0,
        bias=mt.NoBias(),
        step_size=0.55,
        seed=7,
    )
    discrete = mt.MetropolisSampler(
        model,
        space,
        initial=x0,
        bias=mt.DiscreteStateBias.from_space(space, height=0.02),
        step_size=0.55,
        seed=7,
    )
    gaussian = mt.MetropolisSampler(
        model,
        space,
        initial=x0,
        bias=mt.StateConditionedTorsionBias.from_space(
            space,
            height=0.02,
            sigma=0.25,
            n_grid=96,
        ),
        step_size=0.55,
        seed=7,
    )

    for _ in range(n_steps):
        plain.step()
        discrete.step()
        gaussian.step()

    print(f"plain:    n_visited={plain.n_visited:4d} coverage={plain.coverage:.3f}")
    print(f"discrete: n_visited={discrete.n_visited:4d} coverage={discrete.coverage:.3f}")
    print(f"gaussian: n_visited={gaussian.n_visited:4d} coverage={gaussian.coverage:.3f}")

    plt.figure()
    plt.plot(plain.steps, plain.n_unique_by_step, label="plain MC")
    plt.plot(discrete.steps, discrete.n_unique_by_step, label="MC + DiscreteStateBias")
    plt.plot(gaussian.steps, gaussian.n_unique_by_step, label="MC + StateConditionedTorsionBias")
    plt.xlabel("MC step")
    plt.ylabel("Unique global states visited")
    plt.title("Torsion-chain state-space coverage")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
