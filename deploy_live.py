# deploy_live.py
import torch
import numpy as np
import time
import serial # مكتبة الاتصال بالـ ESP32

from architecture import Metric_ContinualNet
from continual_algorithms import StreamingLDA
from haptic_alif import ALIF_Encoder

# ==========================================
# دوال التحليل البيولوجي والذكاء الاصطناعي
# ==========================================
def calculate_cocontraction(flexor_signal, extensor_signal, epsilon=1e-5):
    """
    حساب مؤشر التصلب المشترك لقراءة نية المريض (هل هو حذر؟)
    """
    flex_env = np.mean(np.abs(flexor_signal))
    ext_env = np.mean(np.abs(extensor_signal))
    
    overlap = 2 * min(flex_env, ext_env)
    total_activation = flex_env + ext_env + epsilon
    
    return overlap / total_activation

def enable_dropout(model):
    """تفعيل طبقات الـ Dropout أثناء الاستنتاج لقياس الشك"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()

def live_inference_step(model, slda, raw_sensor_input, uncertainty_threshold=0.1, num_passes=5, device='cuda'):
    """خطوة الذكاء الاصطناعي الأساسية للتنبؤ بالحركة والقوة"""
    model.eval()
    x = torch.tensor(raw_sensor_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        enable_dropout(model)
        pass_preds = []
        for _ in range(num_passes):
            embeddings, force = model(x)
            pred_class, _ = slda.predict(embeddings)
            pass_preds.append(pred_class.item())
        
        pass_preds_tensor = torch.tensor(pass_preds, dtype=torch.float32)
        uncertainty = torch.var(pass_preds_tensor).item()
        final_pred = torch.mode(pass_preds_tensor).values.item()
        force_val = force.item()
        
        if uncertainty > uncertainty_threshold:
            print("⚠️ [Safety Triggered] High Uncertainty! Forcing Rest (0).")
            return 0, 0.0
            
        print(f"✅ Class: {int(final_pred)} | Force: {force_val:.2f} | Unc: {uncertainty:.4f}")
        return int(final_pred), force_val

def read_emg_window_from_serial(ser_port, window_size=200, num_channels=8):
    """
    دالة مساعدة لقراءة نافذة كاملة من الإشارات (200 قراءة) من الـ ESP32
    (في حالة التشغيل الوهمي للمحاكاة، سترجع إشارة عشوائية)
    """
    if ser_port is None:
        return np.random.randn(window_size, num_channels)
    
    # هنا يتم تجميع 200 سطر من السيريال لتكوين نافذة للإدخال
    # (هذا تبسيط للكود، يمكن تحسينه لاحقاً حسب شكل طباعة الـ ESP32)
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
# نقطة التشغيل الأساسية للهاردوير (Live Loop)
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Initializing Live Edge System...")
    
    # 1. إعداد الاتصال بالـ ESP32
    # ملحوظة: غيري 'COM3' لاسم البورت الخاص بك (مثلاً 'COM5' في الويندوز أو '/dev/ttyUSB0' في لينكس)
    SERIAL_PORT_NAME = 'COM3' 
    BAUD_RATE = 115200
    try:
        esp32_serial = serial.Serial(SERIAL_PORT_NAME, BAUD_RATE, timeout=1)
        print(f"🔗 Serial Connection Established on {SERIAL_PORT_NAME}.")
    except Exception as e:
        print(f"⚠️ Serial port {SERIAL_PORT_NAME} not found. Running in SIMULATION mode.")
        esp32_serial = None

    # 2. تحميل الذكاء الاصطناعي والعصب الحسابي
    model = Metric_ContinualNet(embedding_dim=32).to(device)
    try:
        model.load_state_dict(torch.load("trained_doa_backbone.pth", map_location=device))
        print("✅ Loaded backbone weights.")
    except:
        print("⚠️ Weights not found. Using untrained model for demo.")
    
    slda = StreamingLDA(input_dim=32, device=device)
    haptic_neuron = ALIF_Encoder(base_threshold=0.5)
    print("✅ ALIF Haptic Neuron Initialized.")
    
    print("⏳ Waiting for Live EMG Feed...\n")
    print("="*50)
    
    # 3. دورة التشغيل الحية (While True)
    try:
        while True: # تعمل باستمرار لحين إيقاف البرنامج يدوياً
            
            # أ. استقبال الإشارة من الـ ESP32
            live_signal = read_emg_window_from_serial(esp32_serial, window_size=200, num_channels=8)
            
            # ب. تحليل نية المريض (التصلب العضلي)
            flexor_channel = live_signal[:, 0] # بافتراض القناة 0 هي القابضة
            extensor_channel = live_signal[:, 1] # بافتراض القناة 1 هي الباسطة
            stiffness = calculate_cocontraction(flexor_channel, extensor_channel)
            
            stiffness_flag = 1 if stiffness > 0.6 else 0
            if stiffness_flag == 1:
                print("🧠 [Intent] High Co-contraction! (Fragile/Precise Mode)")
                
            try:
                # ج. تشغيل الذكاء الاصطناعي (Class & Force)
                pred, force_val = live_inference_step(model, slda, live_signal, device=device)
                
                # د. تحويل القوة لنبضات الإحساس العكسي (ALIF)
                spike = haptic_neuron.step(force_val)
                if spike == 1:
                    print("   ⚡ BZZZ! Haptic Spike Generated.")
                
                # هـ. إرسال الأوامر المجمعة للـ ESP32
                # شكل حزمة البيانات: (رقم الحركة, قوة المسكة, علامة التصلب, نبضة الهزاز)
                # مثال: "3,0.85,1,1\n"
                command_packet = f"{pred},{force_val:.2f},{stiffness_flag},{spike}\n"
                
                if esp32_serial:
                    esp32_serial.write(command_packet.encode('utf-8'))
                    print(f"   📤 Sent to ESP32: {command_packet.strip()}")
                    
            except Exception as e:
                print(f"❌ Prediction skipped (SLDA likely unfitted): {e}")
                
            # فاصل زمني صغير لتخفيف الحمل على المعالج
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Live Edge System...")
        if esp32_serial:
            esp32_serial.close()
            print("🔌 Serial port closed safely.")