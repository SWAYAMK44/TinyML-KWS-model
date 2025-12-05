import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa

# ------------------------ CONFIG ------------------------
MODEL_PATH = "kws_stop_yes_right_int8.tflite"

CLASSES = ["stop", "yes", "right"]

SAMPLE_RATE = 16000
RECORD_SECONDS = 1

NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

N_FFT = 512
WIN_LENGTH = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.020 * SAMPLE_RATE)
N_MELS = 40
N_MFCC = 13
FMIN = 20
FMAX = 4000


# ------------------- LOAD TFLITE MODEL -------------------
print("Loading INT8 TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Model loaded!")


# -------------------- MFCC EXTRACTION ---------------------
def extract_mfcc_from_audio(audio):

    # Normalize audio (MUST match training)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Pad/trim to 1 second
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        window="hann",
        center=False,
        power=2.0,
        htk=True,
        norm=None
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=N_MFCC,
        dct_type=2,
        norm="ortho"
    )

    mfcc = mfcc.T  # (49,13)
    mfcc = np.expand_dims(mfcc, axis=-1)

    return mfcc.astype(np.float32)


# -------------------- INT8 INFERENCE ----------------------
def run_inference(mfcc):

    # Quantize to INT8 using input scale/zero-point
    scale, zero = input_details["quantization"]
    mfcc_int8 = (mfcc / scale + zero).astype(np.int8)

    mfcc_int8 = np.expand_dims(mfcc_int8, axis=0)  # batch=1

    interpreter.set_tensor(input_details["index"], mfcc_int8)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])[0]

    # Dequantize scores
    out_scale, out_zero = output_details["quantization"]
    probs = (output.astype(np.float32) - out_zero) * out_scale

    return probs


# -------------------- REAL-TIME LOOP ----------------------
print("--------------------------------------")
print("Press ENTER to record 1 second...")
print("Ctrl+C to exit.")
print("--------------------------------------")

while True:
    input("\n➡ Press ENTER to record...")

    print("🎤 Recording...")
    audio = sd.rec(int(NUM_SAMPLES), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    mfcc = extract_mfcc_from_audio(audio)
    probs = run_inference(mfcc)

    pred_idx = np.argmax(probs)
    pred_word = CLASSES[pred_idx]
    conf = probs[pred_idx]

    print(f"🔍 Prediction: {pred_word} (confidence = {conf:.2f})")
