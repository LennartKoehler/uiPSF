"""
Progress reporting abstraction for the PSF-learning pipeline.

Decouples progress display from business logic so that computational
functions never take wall-clock time as a parameter.  The reporter
tracks cumulative elapsed time across stages internally.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from tqdm import tqdm


class ProgressReporter(ABC):
    """Interface for pipeline progress reporting."""

    @abstractmethod
    def begin_stage(self, name: str, total: int = 0, show_loss: bool = False) -> None:
        """Start a new pipeline stage, closing any previous stage.

        Parameters
        ----------
        name : str
            Human-readable stage label (e.g. ``"3/6: learning"``).
        total : int
            Number of update steps expected (0 = indeterminate).
        show_loss : bool
            Whether to display a ``current loss`` field.
        """

    @abstractmethod
    def update(self, n: int = 1, **metrics: Any) -> None:
        """Advance the current stage by *n* steps."""

    @abstractmethod
    def close(self) -> None:
        """Close the current stage and release resources."""

    @abstractmethod
    def get_elapsed(self) -> float:
        """Return cumulative elapsed seconds across all stages."""


class TqdmProgressReporter(ProgressReporter):
    """Live progress display using tqdm bars with cumulative timing."""

    def __init__(self) -> None:
        self._cumulative_elapsed: float = 0.0
        self._stage_start: Optional[float] = None
        self._pbar: Optional[tqdm] = None
        self._show_loss: bool = False
        self._current_loss: Any = None

    def begin_stage(self, name: str, total: int = 0, show_loss: bool = False) -> None:
        self._close_current()
        self._stage_start = time.monotonic()
        self._show_loss = show_loss
        self._current_loss = None
        self._pbar = tqdm(total=total if total > 0 else None, desc=name)
        self._refresh_display()

    def update(self, n: int = 1, **metrics: Any) -> None:
        if self._pbar is None:
            return
        if "loss" in metrics:
            self._current_loss = metrics["loss"]
        self._refresh_display()
        self._pbar.update(n)

    def close(self) -> None:
        self._close_current()

    def get_elapsed(self) -> float:
        if self._stage_start is not None:
            return self._cumulative_elapsed + (time.monotonic() - self._stage_start)
        return self._cumulative_elapsed

    def _close_current(self) -> None:
        if self._pbar is not None:
            self._cumulative_elapsed += time.monotonic() - self._stage_start
            self._pbar.close()
            self._pbar = None
            self._stage_start = None

    def _refresh_display(self) -> None:
        if self._pbar is None:
            return
        parts: list[str] = []
        if self._show_loss and self._current_loss is not None:
            try:
                parts.append(f"current loss: {float(self._current_loss):.5f}")
            except (TypeError, ValueError):
                parts.append(f"current loss: {self._current_loss}")
        parts.append(f"total time: {self.get_elapsed():.2f}s")
        self._pbar.set_postfix_str(", ".join(parts))


class SilentReporter(ProgressReporter):
    """No-op reporter that only tracks elapsed time."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def begin_stage(self, name: str, total: int = 0, show_loss: bool = False) -> None:
        self._start = time.monotonic()

    def update(self, n: int = 1, **metrics: Any) -> None:
        pass

    def close(self) -> None:
        if self._start is not None:
            self._elapsed += time.monotonic() - self._start
            self._start = None

    def get_elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.monotonic() - self._start)
        return self._elapsed
