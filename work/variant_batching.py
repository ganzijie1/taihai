from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np


_ACTIVE_LAYOUT: VariantBatchLayout | None = None


@dataclass(frozen=True)
class VariantBatchLayout:
    names: tuple[str, ...]
    paths_per_variant: int

    @property
    def variants(self) -> int:
        return len(self.names)

    @property
    def total_paths(self) -> int:
        return self.variants * self.paths_per_variant

    def path_slice(self, variant: int | str) -> slice:
        index = self.names.index(variant) if isinstance(variant, str) else int(variant)
        opening = index * self.paths_per_variant
        return slice(opening, opening + self.paths_per_variant)

    def flatten(self, value: np.ndarray) -> np.ndarray:
        if value.shape[:2] != (self.variants, self.paths_per_variant):
            raise ValueError("variant state must start with [variant, path]")
        return value.reshape((self.total_paths,) + value.shape[2:])

    def unflatten(self, value: np.ndarray) -> np.ndarray:
        if value.shape[0] != self.total_paths:
            raise ValueError("flattened state has an incompatible path axis")
        return value.reshape((self.variants, self.paths_per_variant) + value.shape[1:])

    def broadcast_paths(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.shape[0] != self.paths_per_variant:
            raise ValueError("common-random-number source must start with base path axis")
        return np.repeat(value[None], self.variants, axis=0).reshape(
            (self.total_paths,) + value.shape[1:]
        )


class SharedVariantGenerator:
    """Generator proxy broadcasting path-indexed draws across variants."""

    _PATH_DRAW_METHODS = {
        "beta", "binomial", "choice", "dirichlet", "exponential", "gamma", "integers",
        "logistic", "lognormal", "normal", "poisson", "random", "standard_normal",
        "pareto", "triangular", "uniform", "weibull",
    }
    _SIZE_POSITION = {
        "beta": 2, "binomial": 2, "choice": 1, "dirichlet": 1, "exponential": 1,
        "gamma": 2, "integers": 2, "logistic": 2, "lognormal": 2,
        "normal": 2, "poisson": 1, "random": 0, "standard_normal": 0, "pareto": 1,
        "triangular": 3, "uniform": 2, "weibull": 1,
    }

    def __init__(self, generator: np.random.Generator, layout: VariantBatchLayout):
        self._generator = generator
        self._layout = layout

    def __getattr__(self, name):
        attribute = getattr(self._generator, name)
        if name == "permutation":
            def shared_permutation(value, *args, **kwargs):
                if isinstance(value, (int, np.integer)) and int(value) == self._layout.total_paths:
                    base = attribute(self._layout.paths_per_variant, *args, **kwargs)
                    return np.concatenate([
                        base + variant * self._layout.paths_per_variant
                        for variant in range(self._layout.variants)
                    ])
                if isinstance(value, np.ndarray) and value.shape and value.shape[0] == self._layout.total_paths:
                    base_index = attribute(self._layout.paths_per_variant, *args, **kwargs)
                    grouped = value.reshape(
                        (self._layout.variants, self._layout.paths_per_variant) + value.shape[1:]
                    )
                    return grouped[:, base_index].reshape(value.shape)
                return attribute(value, *args, **kwargs)
            return shared_permutation
        if name not in self._PATH_DRAW_METHODS or not callable(attribute):
            return attribute

        def shared(*args, **kwargs):
            args = list(args)
            size_position = self._SIZE_POSITION[name]
            size = kwargs.get("size")
            size_in_args = size is None and len(args) > size_position
            if size_in_args:
                size = args[size_position]
            leading = size[0] if isinstance(size, tuple) else size
            if leading is None:
                arrays = [value for value in (*args, *kwargs.values()) if isinstance(value, np.ndarray)]
                leading = next((value.shape[0] for value in arrays if value.ndim), None)
            if leading != self._layout.total_paths:
                return attribute(*args, **kwargs)

            def slice_value(value, variant):
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == self._layout.total_paths:
                    return value[self._layout.path_slice(variant)]
                return value

            base_size = (
                (self._layout.paths_per_variant,) + tuple(size[1:])
                if isinstance(size, tuple) else self._layout.paths_per_variant
            )
            opening_state = self._generator.bit_generator.state
            results = []
            closing_state = None
            for variant in range(self._layout.variants):
                self._generator.bit_generator.state = opening_state
                call_args = [slice_value(value, variant) for value in args]
                call_kwargs = {key: slice_value(value, variant) for key, value in kwargs.items()}
                if size is not None:
                    if size_in_args:
                        call_args[size_position] = base_size
                    else:
                        call_kwargs["size"] = base_size
                results.append(attribute(*call_args, **call_kwargs))
                if variant == 0:
                    closing_state = self._generator.bit_generator.state
            self._generator.bit_generator.state = closing_state
            return np.concatenate(results, axis=0)

        return shared


@contextmanager
def shared_variant_random_context(layout: VariantBatchLayout) -> Iterator[None]:
    """Make every newly-created NumPy generator obey the CRN variant axis."""
    global _ACTIVE_LAYOUT
    original = np.random.default_rng
    previous_layout = _ACTIVE_LAYOUT
    _ACTIVE_LAYOUT = layout

    def factory(seed=None):
        return SharedVariantGenerator(original(seed), layout)

    np.random.default_rng = factory
    try:
        yield
    finally:
        np.random.default_rng = original
        _ACTIVE_LAYOUT = previous_layout


def active_variant_layout() -> VariantBatchLayout | None:
    return _ACTIVE_LAYOUT


def tile_frozen_draws(draws: dict[str, np.ndarray], layout: VariantBatchLayout) -> dict[str, np.ndarray]:
    return {key: layout.broadcast_paths(value) for key, value in draws.items()}
