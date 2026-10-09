import numpy as np

class ALIF_Encoder:
    """
    Adaptive Leaky Integrate-and-Fire Neuron for Haptic Feedback
    Converts grip force into electrical pulses (Spikes) for haptic feedback.
    """
    def __init__(self, base_threshold=0.5, tau_m=20.0, tau_th=100.0, adaptation_inc=0.2):
        self.base_threshold = base_threshold
        self.tau_m = tau_m          # Membrane time constant
        self.tau_th = tau_th        # Threshold time constant
        self.adaptation_inc = adaptation_inc

        # State Variables
        self.v_m = 0.0              # Membrane potential
        self.v_th = base_threshold  # Dynamic threshold

    def step(self, force_input, dt=1.0):
        """
        Performs a single step of the ALIF Neuron.
        Args:
            force_input: Grip force (0.0 to 1.0)
            dt: Time step between iterations (in milliseconds)
        Returns:
            spike: 1 if a spike is fired, 0 otherwise
        """
        # 1. Leakage (Membrane potential decay)
        self.v_m = self.v_m * np.exp(-dt / self.tau_m)
        self.v_th = self.base_threshold + (self.v_th - self.base_threshold) * np.exp(-dt / self.tau_th)

        # 2. Integration (Signal accumulation)
        self.v_m += (force_input * 1.5)  # Signal amplification

        # 3. Fire and Reset (Spike generation and state reset)
        if self.v_m >= self.v_th:
            spike = 1
            self.v_m = 0.0 
            self.v_th += self.adaptation_inc  # Adaptation (Threshold increment)
        else:
            spike = 0

        return spike

    def reset_state(self):
        """Resets the state of the Neuron."""
        self.v_m = 0.0
        self.v_th = self.base_threshold