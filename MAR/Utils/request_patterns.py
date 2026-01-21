import time
from typing import Optional

import numpy as np


class RequestPattern:
    def __init__(
        self,
        pattern: str = "poisson",
        rate: float = 5.0,
        spike_intensity: float = 10.0,
        spike_period: float = 20.0,
        burst_duration: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            pattern: {"poisson", "microburst", "sustained"}
            rate: base arrival rate (req/sec)
            spike_intensity: multiplier during microburst
            spike_period: seconds between microbursts
            burst_duration: seconds of burst inside each period
        """
        self.pattern = pattern.lower()
        self.rate = float(rate)
        self.spike_intensity = float(spike_intensity)
        self.spike_period = float(spike_period)
        self.burst_duration = float(burst_duration)
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(seed)

    def next_delay(self) -> float:
        if self.rate <= 0:
            return 0.0
        t = (time.monotonic() - self._t0) % max(self.spike_period, 1e-6)

        if self.pattern == "poisson":
            rate = self.rate
        elif self.pattern == "microburst":
            in_burst = t < self.burst_duration
            rate = self.rate * self.spike_intensity if in_burst else self.rate
            rate *= float(self._rng.uniform(0.9, 1.1))
        elif self.pattern == "sustained":
            rate = self.rate * 3.0
            rate *= float(self._rng.uniform(0.95, 1.05))
        else:
            rate = self.rate

        rate = max(rate, 1e-6)
        return float(self._rng.exponential(1.0 / rate))
