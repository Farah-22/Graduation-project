# deploy_live.py
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
import serial

from architecture import Metric_ContinualNet
from continual_algorithms import StreamingLDA
from dataset_handler import apply_myo_filters
from evaluate import enable_dropout
from haptic_alif import ALIF_Encoder
import config as cfg

# ==========================================
# Biological Analysis Functions
# ==========================================
def calculate_cocontraction(flexor_signal, extensor_signal, epsilon=1e-5):
    """
    Calculate co-contraction index to read patient intent (cautious mode).
    """
    flex_env = np.mean(np.abs(flexor_signal))
    ext_env = np.mean(np.abs(extensor_signal))
    
    overlap = 2 * min(flex_env, ext_env)
    total_activation = flex_env + ext_env + epsilon
    
    return overlap / total_activation

# ==========================================
# AI Inference Function
# ==========================================
def live_inference_step(model, slda, raw_sensor_input, uncertainty_threshold=0.2, num_passes=5, device='cuda'):
    """
    Core AI step: Predict movement class and force with MC Dropout safety.
    """
    model.eval()
    
    # 1. Preprocessing (Filters + Scaler) - THIS WAS MISSING!
    filtered = apply_myo_filters(raw_sensor_input, fs=cfg.SAMPLING_RATE)
    scaled = scaler.transform(filtered)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 2. MC Dropout for uncertainty estimation
        enable_dropout(model)
        pass_preds = []
        force_values = []
        
        for _ in range(num_passes):
            embeddings, force = model(x)
            pred_class, _ = slda.predict(embeddings)
            pass_preds.append(pred_class.item())
            force_values.append(force.item())
        
        # 3. Calculate uncertainty and final prediction
        pass_preds_tensor = torch.tensor(pass_preds, dtype=torch.float32)
        uncertainty = torch.var(pass_preds_tensor).item()
        final_pred = torch.mode(pass_preds_tensor).values.item()
        avg_force = np.mean(force_values)
        
        # 4. Safety check
        if uncertainty > uncertainty_threshold:
            print("⚠️ [Safety Triggered] High Uncertainty! Forcing Rest (0).")
            return 0, 0.0, uncertainty
        
        print(f"✅ Class: {int(final_pred)} | Force: {avg_force:.2f} | Unc: {uncertainty:.4f}")
        return int(final_pred), avg_force, uncertainty

# ==========================================
# Serial Communication with ESP32
# ==========================================
def read_emg_window_from_serial(ser_port, window_size=200, num_channels=8):
    """
    Read a full window (200 samples) from ESP32 serial.
    Falls back to random data if no serial connection (simulation mode).
    """
    if ser_port is None:
        return np.random.randn(window_size, num_channels)
    
    emg_data = []
    while len(emg_data) < window_size:
        try:
            line = ser_port.readline().decode('utf-8').strip()
            if line:
                values = [float(v) for v in line.split(',')]
                if len(values) == num_channels:
                    emg_data.append(values)
        except:
            pass
    return np.array(emg_data)

# ==========================================
# Main Live Loop
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("🚀 Myo Prosthetic Control System - Live Edge")
    print("="*60)
    
    # 1. Load Scaler (CRITICAL - was missing in old version!)
    scaler_path = os.path.join(cfg.WEIGHTS_DIR, 'subject_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Loaded subject scaler from {scaler_path}")
    
    # 2. Load Model (CORRECT weights file!)
    model = Metric_ContinualNet(
        embedding_dim=cfg.EMBEDDING_DIM,
        num_sensors=cfg.NUM_SENSORS,
        window_size=cfg.WINDOW_SIZE
    ).to(device)
    
    model_path = os.path.join(cfg.WEIGHTS_DIR, 'metric_model_myo.pth')
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Loaded model weights from {model_path}")
    except Exception as e:
        print(f"⚠️ Weights not found: {e}")
        print("   Running with untrained model (demo mode only).")
    
    # 3. Load Recalibrated SLDA (NOT empty new one!)
    slda_path = os.path.join(cfg.WEIGHTS_DIR, 'slda_classifier_recalibrated.pkl')
    try:
        with open(slda_path, 'rb') as f:
            slda = pickle.load(f)
        print(f"✅ Loaded recalibrated SLDA from {slda_path}")
    except Exception as e:
        print(f"⚠️ SLDA not found: {e}")
        print("   Creating empty SLDA (will not classify correctly).")
        slda = StreamingLDA(input_dim=cfg.EMBEDDING_DIM, device=device)
    
    # 4. Initialize Haptic Feedback
    haptic_neuron = ALIF_Encoder(base_threshold=0.5)
    print("✅ ALIF Haptic Neuron Initialized.")
    
    # 5. Setup ESP32 Serial Connection
    SERIAL_PORT_NAME = 'COM3'  # Change to your port (e.g., 'COM5' or '/dev/ttyUSB0')
    BAUD_RATE = 115200
    try:
        esp32_serial = serial.Serial(SERIAL_PORT_NAME, BAUD_RATE, timeout=1)
        print(f" Serial Connection Established on {SERIAL_PORT_NAME}.")
    except Exception as e:
        print(f"⚠️ Serial port {SERIAL_PORT_NAME} not found. Running in SIMULATION mode.")
        esp32_serial = None
    
    print("\n⏳ Waiting for Live EMG Feed...\n")
    print("="*60)
    
    # 6. Main Live Loop
    try:
        while True:
            # A. Receive signal from ESP32
            live_signal = read_emg_window_from_serial(
                esp32_serial, 
                window_size=cfg.WINDOW_SIZE, 
                num_channels=cfg.NUM_SENSORS
            )
            
            # B. Analyze patient intent (muscle stiffness)
            flexor_channel = live_signal[:, 0]   # Assuming channel 0 is flexor
            extensor_channel = live_signal[:, 1] # Assuming channel 1 is extensor
            stiffness = calculate_cocontraction(flexor_channel, extensor_channel)
            
            stiffness_flag = 1 if stiffness > 0.6 else 0
            if stiffness_flag == 1:
                print(" [Intent] High Co-contraction! (Fragile/Precise Mode)")
            
            try:
                # C. Run AI inference (Class + Force + Uncertainty)
                pred, force_val, uncertainty = live_inference_step(
                    model, slda, live_signal, 
                    uncertainty_threshold=cfg.UNCERTAINTY_THRESHOLD,
                    num_passes=cfg.MC_DROPOUT_PASSES,
                    device=device
                )
                
                # D. Convert force to haptic feedback pulses (ALIF)
                spike = haptic_neuron.step(force_val)
                if spike == 1:
                    print("   ⚡ BZZZ! Haptic Spike Generated.")
                
                # E. Send command packet to ESP32
                # Format: (movement_class, grip_force, stiffness_flag, haptic_spike)
                # Example: "3,0.85,1,1\n"
                command_packet = f"{pred},{force_val:.2f},{stiffness_flag},{spike}\n"
                
                if esp32_serial:
                    esp32_serial.write(command_packet.encode('utf-8'))
                    print(f"    Sent to ESP32: {command_packet.strip()}")
                    
            except Exception as e:
                print(f"❌ Prediction skipped: {e}")
            
            # Small delay to reduce CPU load
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Live Edge System...")
        if esp32_serial:
            esp32_serial.close()
            print("🔌 Serial port closed safely.")