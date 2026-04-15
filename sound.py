import numpy as np
from scipy.io import wavfile

def generate_piano_note(filename, frequency, duration=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a sine wave with a simple fade-out (envelope) to sound like a piano
    note = np.sin(2 * np.pi * frequency * t) * np.exp(-3 * t) 
    # Normalize and convert to 16-bit PCM
    audio = (note * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio)

# Frequencies for the C4 Major scale
notes = {
    "C.wav": 261.63,
    "D.wav": 293.66,
    "E.wav": 329.63,
    "F.wav": 349.23,
    "G.wav": 392.00,
    "A.wav": 440.00,
    "B.wav": 493.88
}

print("Generating sound files...")
for name, freq in notes.items():
    generate_piano_note(name, freq)
    print(f"Created {name}")
print("Done! You now have all the sounds needed.")
