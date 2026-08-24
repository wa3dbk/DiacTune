from __future__ import annotations

from abc import ABC, abstractmethod


class DiacritizationBackend(ABC):
    @abstractmethod
    def infer(self, sentences: list[str]) -> list[str]: ...

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        **kwargs,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning")

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")


_REGISTRY: dict[str, str] = {
    "camel":  "diac.backends.camel.CAMeLBackend",
    "byt5":   "diac.backends.byt5.ByT5Backend",
    "rababa": "diac.backends.rababa.RababaBackend",
    "catt":   "diac.backends.catt.CATTBackend",
}


def get_backend(name: str) -> DiacritizationBackend:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
