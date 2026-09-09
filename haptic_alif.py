# haptic_alif.py
import numpy as np

class ALIF_Encoder:
    def __init__(self, base_threshold=0.5, tau_m=20.0, tau_th=100.0, adaptation_inc=0.2):
        
        self.base_threshold = base_threshold
        self.tau_m = tau_m
        self.tau_th = tau_th
        self.adaptation_inc = adaptation_inc

        # State Variables (
        self.v_m = 0.0                   
        self.v_th = base_threshold       

    def step(self, force_input, dt=1.0):
        
        # 1. Leakage 
        self.v_m = self.v_m * np.exp(-dt / self.tau_m)
        self.v_th = self.base_threshold + (self.v_th - self.base_threshold) * np.exp(-dt / self.tau_th)

        # 2. Integration 
        
        self.v_m += (force_input * 1.5)

        # 3. Fire and Reset 
        if self.v_m >= self.v_th:
            spike = 1
            self.v_m = 0.0 
            self.v_th += self.adaptation_inc #(Adaptation)
        else:
            spike = 0

        return spike

    def reset_state(self):
       
        self.v_m = 0.0
        self.v_th = self.base_threshold