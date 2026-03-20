"""
SOT (Spin-Orbit Torque) MTJ device model.

Models a magnetic tunnel junction (MTJ) controlled via spin-orbit torque.
Uses the Neel-Brown thermally-activated switching model.

Physical parameters:
- Resistance: P = 1 kΩ, AP = 2 kΩ  (TMR ~ 100%)
- Thermal stability factor: Eb/kT ~ 40
- Critical switching current: Ic0 ~ 100 µA
- Attempt frequency: 1/tau0 ~ 1 GHz (tau0 = 1 ns)
- Switching energy: ~10 pJ per event
"""

import math
import random


class SOTDevice:
    """
    Magnetic Tunnel Junction (MTJ) device driven by Spin-Orbit Torque.

    States:
        'parallel'      (P)  → low resistance, high conductance  → logical 0
        'antiparallel'  (AP) → high resistance, low conductance  → logical 1

    Neel-Brown switching probability model:
        P_sw = 1 - exp(-t_pulse / (tau0 * exp(E_b/kT * (1 - I/I_c0))))

    Args:
        state (str): Initial state — 'parallel' or 'antiparallel'.
    """

    # Physical constants
    R_PARALLEL_OHM = 1_000       # 1 kΩ
    R_ANTIPARALLEL_OHM = 2_000   # 2 kΩ  (TMR ~ 100%)
    E_B_OVER_KT = 40.0           # Thermal stability factor (dimensionless)
    I_C0_UA = 100.0              # Critical switching current (µA)
    TAU0_NS = 1.0                # Attempt period (ns)
    SWITCH_ENERGY_PJ = 10.0      # Energy per switching event (pJ)

    def __init__(self, state: str = "parallel"):
        if state not in ("parallel", "antiparallel"):
            raise ValueError("state must be 'parallel' or 'antiparallel'")
        self._state = state

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current MTJ state: 'parallel' or 'antiparallel'."""
        return self._state

    @property
    def is_parallel(self) -> bool:
        return self._state == "parallel"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def conductance(self) -> float:
        """Return device conductance in Siemens (S)."""
        r = self.R_PARALLEL_OHM if self.is_parallel else self.R_ANTIPARALLEL_OHM
        return 1.0 / r

    def switching_probability(self, current_ua: float, pulse_ns: float = 1.0) -> float:
        """
        Compute Neel-Brown switching probability.

        Args:
            current_ua: Applied current in µA.
            pulse_ns:   Pulse duration in nanoseconds.

        Returns:
            Probability of switching [0, 1].
        """
        if current_ua <= 0:
            return 0.0

        # Thermal stability is reduced by applied current
        ratio = current_ua / self.I_C0_UA
        if ratio >= 1.0:
            # Above critical current → deterministic switch
            return 1.0

        # Effective barrier: Eb_eff/kT = Eb/kT * (1 - I/Ic0)
        eb_eff = self.E_B_OVER_KT * (1.0 - ratio)

        # Effective attempt time including barrier
        tau_eff_ns = self.TAU0_NS * math.exp(eb_eff)

        prob = 1.0 - math.exp(-pulse_ns / tau_eff_ns)
        return max(0.0, min(1.0, prob))

    def switch(self, current_ua: float, pulse_ns: float = 1.0) -> bool:
        """
        Attempt to switch the MTJ state stochastically.

        Args:
            current_ua: Applied SOT current in µA.
            pulse_ns:   Pulse duration in nanoseconds.

        Returns:
            True if the device switched state; False otherwise.
        """
        p = self.switching_probability(current_ua, pulse_ns)
        if random.random() < p:
            self._state = "antiparallel" if self.is_parallel else "parallel"
            return True
        return False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """Directly set the MTJ state (used for deterministic programming)."""
        if state not in ("parallel", "antiparallel"):
            raise ValueError("state must be 'parallel' or 'antiparallel'")
        self._state = state

    def __repr__(self) -> str:
        g = self.conductance()
        return (
            f"SOTDevice(state={self._state!r}, "
            f"G={g*1e3:.3f} mS, R={1/g:.0f} Ω)"
        )
