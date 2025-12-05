import numpy as np
import sounddevice as sd
import tensorflow as tf
import time
import os

# ==== CONFIG ====
tflite_path = r"C:\Users\swayam\Desktop\KWS_Stop_Project\stop_kws_model_fixed.tflite"
class_labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

SAMPLE_RATE = 16000
WINDOW_SIZE = 16000      # 1 second
STEP_SIZE = 8000         # 0.5 second
CHANNELS = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==== SAME MFCC FUNCTION ====
def compute_mfcc(audio, sample_rate=16000):
    frame_length = 640
    frame_step = 320

    stft = tf.signal.stft(audio, frame_length=frame_length,
                          frame_step=frame_step, fft_length=frame_length)

    spectrogram = tf.abs(stft)

    num_mel_bins = 40
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], sample_rate, 20.0, 4000.0)

    mel_spectrogram = tf.matmul(spectrogram, mel_matrix)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]
    return mfccs


# ==== LOAD TFLITE MODEL ====
print("🔄 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded!")
print("Input shape:", input_details[0]["shape"])

# ==== INFERENCE FUNCTION WITH TIMING ====
def infer_window(audio_window):
    total_start = time.time()

    # --- Preprocessing (MFCC) ---
    pre_start = time.time()
    mfccs = compute_mfcc(audio_window)
    pre_time = (time.time() - pre_start) * 1000

    x = tf.expand_dims(mfccs, axis=0)
    x = tf.expand_dims(x, axis=-1)
    x = x.numpy().astype(np.float32)

    # --- Inference ---
    inf_start = time.time()
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    inf_time = (time.time() - inf_start) * 1000

    # --- Total ---
    total_time = (time.time() - total_start) * 1000

    output = interpreter.get_tensor(output_details[0]["index"])
    idx = np.argmax(output[0])
    conf = float(output[0][idx])
    word = class_labels[idx]

    return word, conf, pre_time, inf_time, total_time


# ==== REAL-TIME AUDIO STREAM =====
buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
stream.start()

print("\n🎤 Listening in real time for TFLite model... (Ctrl+C to stop)\n")

try:
    while True:
        audio_chunk, _ = stream.read(STEP_SIZE)
        audio_chunk = audio_chunk.flatten()

        buffer = np.roll(buffer, -STEP_SIZE)
        buffer[-STEP_SIZE:] = audio_chunk

        word, conf, pre_t, inf_t, total_t = infer_window(buffer)

        if conf > 0.50:
            print(f"{time.strftime('%H:%M:%S')} | {word.upper():>8s} | "
                  f"Conf: {conf:.2f} | Pre: {pre_t:.2f} ms | Inf: {inf_t:.2f} ms | Total: {total_t:.2f} ms")
        else:
            print(f"{time.strftime('%H:%M:%S')} | Listening... "
                  f"(Conf: {conf:.2f}) | Pre: {pre_t:.2f} ms | Inf: {inf_t:.2f} ms | Total: {total_t:.2f} ms")

except KeyboardInterrupt:
    print("\n🛑 Stopped listening.")
    stream.stop()
    stream.close() 

