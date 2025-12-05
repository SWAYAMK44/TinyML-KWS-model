import numpy as np
import sounddevice as sd
import tensorflow as tf
import time
import os

# ==== CONFIG ====
model_path = r"C:\Users\swayam\Desktop\KWS_Stop_Project\stop_kws_model_improved.h5"
class_labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

SAMPLE_RATE = 16000
WINDOW_SIZE = 16000         # 1s window (samples)
STEP_SIZE = 8000            # 0.5s step for rolling window (samples)
CHANNELS = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def compute_mfcc(audio, sample_rate=16000):
    frame_length = 640
    frame_step = 320

    stft = tf.signal.stft(audio, frame_length=frame_length,
                          frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    num_mel_bins = 40
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], sample_rate, 20.0, 4000.0)

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :13]
    return mfccs


def infer_window(audio_window, model):
    # ---------------- TIMING: PREPROCESSING ----------------
    t_pre_start = time.time()

    mfccs = compute_mfcc(audio_window)

    mfcc_input = tf.expand_dims(mfccs, axis=0)
    mfcc_input = tf.expand_dims(mfcc_input, axis=-1)

    t_pre_end = time.time()
    preprocess_time = (t_pre_end - t_pre_start) * 1000  # ms

    # ---------------- TIMING: INFERENCE ----------------
    t_inf_start = time.time()
    preds = model.predict(mfcc_input, verbose=0)
    t_inf_end = time.time()
    inference_time = (t_inf_end - t_inf_start) * 1000  # ms

    # ---------------- RESULTS ----------------
    idx = np.argmax(preds[0])
    conf = float(preds[0][idx])
    word = class_labels[idx]

    return word, conf, preprocess_time, inference_time


# Load the model once
print("🔄 Loading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!")

# Start mic stream
buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
stream.start()

print("\n🎤 Listening in real time. Speak a command! (Press Ctrl+C to stop)\n")

try:
    while True:
        audio_chunk, _ = stream.read(STEP_SIZE)
        audio_chunk = audio_chunk.flatten()

        buffer = np.roll(buffer, -STEP_SIZE)
        buffer[-STEP_SIZE:] = audio_chunk

        t_total_start = time.time()
        word, conf, pre_t, inf_t = infer_window(buffer, model)
        t_total_end = time.time()
        total_time = (t_total_end - t_total_start) * 1000  # ms

        if conf > 0.5:
            print(f"{time.strftime('%H:%M:%S')} | {word.upper():>8s} | "
                  f"Conf: {conf:.2f} | Pre: {pre_t:.2f} ms | Inf: {inf_t:.2f} ms | Total: {total_time:.2f} ms")
        else:
            print(f"{time.strftime('%H:%M:%S')} | Listening... (Conf: {conf:.2f}) | "
                  f"Pre: {pre_t:.2f} ms | Inf: {inf_t:.2f} ms | Total: {total_time:.2f} ms")

except KeyboardInterrupt:
    print("\n🛑 Stopped listening.")
    stream.stop()
    stream.close()
