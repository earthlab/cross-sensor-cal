"""Utilities for consistent tile-based progress reporting."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TileProgressReporter:
    """Unified progress reporter for tile-based loops."""

    def __init__(
        self,
        stage_name: str,
        total_tiles: int,
        interactive_mode: bool,
        log_every: int = 25,
    ) -> None:
        self.stage_name = stage_name
        self.total = total_tiles
        self.interactive = interactive_mode
        self.log_every = max(1, log_every)

        self.count = 0
        self._tqdm: Optional[object] = None

        if self.interactive:
            from tqdm import tqdm  # type: ignore[import-not-found]

            self._tqdm = tqdm(
                total=self.total,
                desc=stage_name,
                unit="tile",
                leave=False,
                mininterval=0.5,
            )

    def update(self, n: int = 1) -> None:
        self.count += n
        if self.interactive and self._tqdm is not None:
            self._tqdm.update(n)  # type: ignore[call-arg]
        else:
            if (self.count % self.log_every == 0) or (self.count == self.total):
                pct = (self.count / self.total) * 100 if self.total > 0 else 0.0
                logger.info(
                    "%s progress: %d/%d tiles (%.1f%%)",
                    self.stage_name,
                    self.count,
                    self.total,
                    pct,
                )

    def close(self) -> None:
        if self.interactive and self._tqdm is not None:
            self._tqdm.close()  # type: ignore[call-arg]
