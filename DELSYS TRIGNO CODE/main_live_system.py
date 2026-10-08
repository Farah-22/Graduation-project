import torch
import numpy as np
import time
import serial
import pickle
from pathlib import Path

# Import components
from src.data_processing import apply_emg_filters, apply_lowpass_envelope
from src.models import Metric_ContinualNet, StreamingLDA
from haptic_alif import ALIF_Encoder

# ==========================================
# System Settings
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
MODEL_PATH = WEIGHTS_DIR / "doa_backbone.pth"
SLDA_PATH = WEIGHTS_DIR / "slda_classifier.pkl"
CONFIG_PATH = WEIGHTS_DIR / "pipeline_config.pt"

# ESP32 Connection Settings
SERIAL_PORT = "COM3"  # Change this to your port (e.g., 'COM5' on Windows or '/dev/ttyUSB0' on Linux)
BAUD_RATE = 115200

# Processing Settings
WINDOW_SIZE = 400      # Number of samples in the window
STEP_SIZE = 40         # Step between windows
NUM_CHANNELS = 12      # Number of channels (EMG sensors)
SAMPLING_RATE = 2000   # Sampling rate (Hz)

# Safety Settings
UNCERTAINTY_THRESHOLD = 0.1
MC_DROPOUT_PASSES = 5
MAJORITY_VOTE_WINDOW = 20  # Window size for smoothing

print("="*60)
print(" PROSTHETIC AI - LIVE DEPLOYMENT SYSTEM")
print("="*60)

# ==========================================
# 1. Load Models and Weights
# ==========================================
print("\n[1/4] Loading AI Models...")

try:
    # A) Load the base model (Backbone)
    model = Metric_ContinualNet(
        embedding_dim=32, 
        proj_dim=128, 
        num_sensors=NUM_CHANNELS
    ).to(DEVICE)
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded backbone from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found! Using untrained model.")
    
    model.to(DEVICE)
    model.eval()
    
    # B) Load the SLDA Classifier
    if SLDA_PATH.exists():
        with open(SLDA_PATH, 'rb') as f:
            slda = pickle.load(f)
        print(f"✅ Loaded SLDA classifier from {SLDA_PATH}")
    else:
        print(f"⚠️ Warning: {SLDA_PATH} not found! Creating new SLDA.")
        slda = StreamingLDA(input_dim=32, device=DEVICE)
    
    # C) Initialize Haptic Feedback (ALIF Neuron)
    haptic_encoder = ALIF_Encoder(base_threshold=0.5)
    print("✅ ALIF Haptic Encoder initialized")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# ==========================================
# 2. Connect to ESP32 (Hardware)
# ==========================================
print("\n[2/4] Establishing Hardware Connection...")

try:
    esp32_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for connection to stabilize
    print(f"✅ Connected to ESP32 on {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Serial port {SERIAL_PORT} not available. Running in SIMULATION mode.")
    esp32_serial = None

# ==========================================
# 3. Helper Functions
# ==========================================

def enable_dropout(model):
    """Enable Dropout layers during inference to measure uncertainty."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def calculate_cocontraction(flexor_signal, extensor_signal, epsilon=1e-5):
    """
    Calculate the Co-contraction Index.
    Indicates patient intent: cautious/precise mode.
    """
    flex_env = np.mean(np.abs(flexor_signal))
    ext_env = np.mean(np.abs(extensor_signal))
    overlap = 2 * min(flex_env, ext_env)
    total_activation = flex_env + ext_env + epsilon
    return overlap / total_activation

def mc_dropout_predict(model, slda, x, num_passes=MC_DROPOUT_PASSES):
    """
    Predict using MC Dropout to measure uncertainty.
    
    Returns:
        final_pred: Final predicted movement
        uncertainty: Degree of uncertainty
        force_val: Predicted grip force from AI
    """
    enable_dropout(model)
    pass_preds = []
    force_vals = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            _, embeddings, force = model(x)
            pred_class, _ = slda.predict(embeddings)
            pass_preds.append(pred_class.item())
            force_vals.append(force.item())
    
    # Calculate uncertainty (Variance)
    pass_preds_tensor = torch.tensor(pass_preds, dtype=torch.float32)
    uncertainty = torch.var(pass_preds_tensor).item()
    
    # Majority Vote
    final_pred = int(torch.mode(pass_preds_tensor).values.item())
    
    # Average predicted force
    force_val = np.mean(force_vals)
    
    return final_pred, uncertainty, force_val

def read_emg_window_simulation(window_size=WINDOW_SIZE, num_channels=NUM_CHANNELS):
    """
    Simulate reading a window of signals (for testing without hardware).
    In a real setup, replace this with Delsys or OpenBCI reading.
    """
    # Random signal (simulation)
    return np.random.randn(window_size, num_channels) * 0.1

def read_emg_window_from_serial(ser_port, window_size=WINDOW_SIZE, num_channels=NUM_CHANNELS):
    """
    Read a full window from the ESP32 via Serial.
    """
    if ser_port is None:
        return read_emg_window_simulation(window_size, num_channels)
    
    emg_data = []
    # print("📡 Reading EMG window from serial...") # Commented out to reduce console spam
    
    while len(emg_data) < window_size:
        try:
            line = ser_port.readline().decode('utf-8').strip()
            if line:
                values = [float(v) for v in line.split(',')]
                if len(values) == num_channels:
                    emg_data.append(values)
        except Exception:
            pass
    
    return np.array(emg_data)

def read_fsr_force(ser_port):
    """
    Read the ACTUAL physical grip force from the FSR (Force Sensitive Resistor) 
    sensors placed on the prosthetic fingers.
    
    Note: Adjust the parsing logic based on how your ESP32 firmware formats the FSR data.
    Example expected format from ESP32: "FSR:0.85\n"
    """
    if ser_port is None:
        # Simulate FSR reading (random force between 0.0 and 1.0)
        return np.random.uniform(0.0, 1.0) 
    
    try:
        # Non-blocking read attempt for FSR data
        # Ensure your ESP32 sends FSR data in a distinguishable format
        line = ser_port.readline().decode('utf-8').strip()
        if line.startswith("FSR:"):
            fsr_value = float(line.split(":")[1])
            return np.clip(fsr_value, 0.0, 1.0)
    except Exception:
        pass
    
    # Return 0.0 if no FSR data is received (no contact)
    return 0.0

# ==========================================
# 4. Live Execution Loop (Main Loop)
# ==========================================
print("\n[3/4] System Ready!")
print("[4/4] Starting Live Inference Loop...")
print("="*60)
print("Press Ctrl+C to stop\n")

# Buffer for predictions smoothing (Majority Vote)
prediction_buffer = []

try:
    while True:
        # ⏱️ Loop timing
        loop_start = time.time()
        
        # 1️⃣ Read EMG signal from sensors
        live_emg = read_emg_window_from_serial(esp32_serial)
        
        # 2️⃣ Signal processing (Filtering)
        clean_emg = apply_emg_filters(live_emg, fs=SAMPLING_RATE)
        clean_emg = apply_lowpass_envelope(clean_emg, fs=SAMPLING_RATE, cutoff=10.0)
        
        # Convert to Tensor
        tensor_x = torch.tensor(clean_emg, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 3️⃣ Calculate Co-contraction (Patient Intent)
        # Assuming channel 0 = Flexor, channel 1 = Extensor
        flexor_channel = live_emg[:, 0]
        extensor_channel = live_emg[:, 1]
        stiffness_index = calculate_cocontraction(flexor_channel, extensor_channel)
        stiffness_flag = 1 if stiffness_index > 0.6 else 0
        
        if stiffness_flag == 1:
            print("🧠 [Intent] High co-contraction detected (Precise mode)")
        
        # 4️⃣ AI Prediction (MC Dropout + SLDA)
        try:
            pred_class, uncertainty, predicted_force = mc_dropout_predict(
                model, slda, tensor_x, num_passes=MC_DROPOUT_PASSES
            )
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            continue
        
        # 5️⃣ Safety System: Reject movement if uncertainty is high
        if uncertainty > UNCERTAINTY_THRESHOLD:
            print(f"️ [SAFETY] High uncertainty ({uncertainty:.4f})! Forcing REST.")
            safe_action = 0
            predicted_force = 0.0
        else:
            # Add to predictions buffer for smoothing
            prediction_buffer.append(pred_class)
            if len(prediction_buffer) > MAJORITY_VOTE_WINDOW:
                prediction_buffer.pop(0)
            
            # Majority Vote
            safe_action = int(np.bincount(prediction_buffer).argmax())
        
        # 6️⃣ Read ACTUAL physical force from FSR sensors on the prosthetic fingers
        actual_fsr_force = read_fsr_force(esp32_serial)
        
        # 7️⃣ Generate haptic feedback pulses based on ACTUAL physical touch (FSR)
        # This creates a true closed-loop haptic feedback system
        haptic_spike = haptic_encoder.step(actual_fsr_force, dt=10.0)  # dt=10ms
        
        # 8️⃣ Send commands to ESP32
        # Packet format: (action, predicted_force, stiffness, haptic_spike)
        command_packet = f"{safe_action},{predicted_force:.2f},{stiffness_flag},{haptic_spike}\n"
        
        if esp32_serial:
            try:
                esp32_serial.write(command_packet.encode('utf-8'))
            except Exception as e:
                print(f"❌ Serial send error: {e}")
        
        # 9️⃣ Display results
        print(f"✅ Action: {safe_action} | AI Force: {predicted_force:.2f} | "
              f"FSR Force: {actual_fsr_force:.2f} | Unc: {uncertainty:.3f} | "
              f"Haptic: {'' if haptic_spike else '-'}")
        
        # ⏳ Control loop speed
        loop_time = time.time() - loop_start
        sleep_time = max(0, 0.05 - loop_time)  # 50ms cycle
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n" + "="*60)
    print("🛑 Shutting down system...")
    
    if esp32_serial:
        esp32_serial.close()
        print("🔌 Serial port closed")
    
    print("✅ System stopped safely")