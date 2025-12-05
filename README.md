<img width="1503" height="835" alt="Screenshot 2025-12-05 231500" src="https://github.com/user-attachments/assets/bf69d939-f41e-46ca-9f98-81440d7c88bb" /># TinyML-KWS-model
This repository contains an end-to-end TinyML Keyword Spotting system designed for microcontrollers such as the raspberry pi 4 or even an ESP32-S3 .
The project includes dataset preprocessing, MFCC extraction, training scripts, multiple TFLite models, and real-time inference code for both PC and embedded devices.

Features

MFCC-based audio preprocessing
FP32 and INT8 TensorFlow Lite models
Real-time inference (Python & ESP32 C++)
Memory-optimized architecture for microcontrollers
Works with simple I2S/PDM microphones

Keywords Detected
Yes,
Stop,
Right/

But this can be trained to predict any word in any language. The entire pipeline remains the same, just the dataset needed for model training changes.

Model Input-
49 frames × 40 MFCC coefficients,
Sample rate: 16 kHz,
Window & hop: 25 ms / 10 ms.

How to run inference-
Just run the .py file and make sure to change the model path as needed.

Applications-
One of the very important application is to trigger the emergency stop buttons to stop the escalator with just a trigger word. This can be life saving in some cases
It also has a lot of other IoT applications. This is used in Alexa,siri etc. 

More about this repositry-
This repo has three different model files-
A normal tensoreflow (.h5) file
An optimised tflite float32 file (.tflite)
An optimised tflite int8 file (.tflite)
Each file has it's own script to run inference. that script contains the pre-processing part as well. 

Inference on Raspberry pi4 is almost similar-
Just the preprocessing part should be modified slightly depending on what microphone is being used. 
