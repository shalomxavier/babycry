import numpy as np
import soundfile as sf
import os

def generate_white_noise(duration_seconds, sample_rate, output_file):
    """
    Generate white noise and save it to a .wav file.
    
    Parameters:
    - duration_seconds: Duration of the noise in seconds
    - sample_rate: Sample rate of the audio
    - output_file: Path to the output .wav file
    """
    # Generate white noise
    num_samples = duration_seconds * sample_rate
    noise = np.random.normal(0, 1, num_samples)
    
    # Save to .wav file
    sf.write(output_file, noise, sample_rate)

# Directory to save noise files
output_dir = 'noise'
os.makedirs(output_dir, exist_ok=True)

# Parameters
duration = 5  # Duration of each noise file in seconds
sample_rate = 44100  # Sample rate in Hz

# Generate and save multiple noise files
for i in range(10):  # Create 10 noise files
    output_file = os.path.join(output_dir, f'noise_{i+1}.wav')
    generate_white_noise(duration, sample_rate, output_file)
    print(f'Saved {output_file}')
