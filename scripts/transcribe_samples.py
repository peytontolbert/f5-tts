import os
import glob
from pathlib import Path
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import logging
import librosa
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperTranscriber:
    def __init__(self, model_size="small"):
        logging.info(f"Loading Whisper {model_size} model...")
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.target_sr = 16000  # Whisper requires 16kHz audio
        logging.info(f"Model loaded and running on {self.device}")

    def transcribe_audio(self, audio_path):
        """Transcribe a single audio file"""
        try:
            # Load and resample audio using librosa
            audio, orig_sr = librosa.load(audio_path, sr=None)  # Load at original SR
            if orig_sr != self.target_sr:
                logging.info(f"Resampling audio from {orig_sr}Hz to {self.target_sr}Hz")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
            
            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            
            # Process audio with Whisper processor
            input_features = self.processor(
                audio, 
                sampling_rate=self.target_sr, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logging.error(f"Error transcribing {audio_path}: {str(e)}")
            return None

def transcribe_samples(wavs_dir):
    """Transcribe all WAV files in the directory and create samples.txt"""
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(model_size="small")
    
    # Get all WAV files
    wav_files = glob.glob(os.path.join(wavs_dir, "*.wav"))
    
    if not wav_files:
        logging.warning(f"No WAV files found in {wavs_dir}")
        return
    
    logging.info(f"Found {len(wav_files)} WAV files. Starting transcription...")
    
    # Create samples.txt in the parent directory
    samples_file = os.path.join(os.path.dirname(wavs_dir), "samples.txt")
    
    with open(samples_file, "w", encoding="utf-8") as f:
        for wav_file in wav_files:
            logging.info(f"Transcribing: {os.path.basename(wav_file)}")
            
            transcription = transcriber.transcribe_audio(wav_file)
            if transcription:
                # Write to samples.txt
                f.write(f"{wav_file}|{transcription}\n")
                logging.info(f"✓ Transcribed: {transcription[:50]}...")
            else:
                logging.error(f"Failed to transcribe: {wav_file}")
    
    # Delete any .npy files in the directory
    npy_files = glob.glob(os.path.join(wavs_dir, "*.npy"))
    for npy_file in npy_files:
        try:
            os.remove(npy_file)
            logging.info(f"Deleted: {npy_file}")
        except Exception as e:
            logging.error(f"Error deleting {npy_file}: {str(e)}")

def main():
    # Get the absolute path to voice_profiles/wavs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wavs_dir = os.path.join(script_dir, "voice_profiles", "wavs")
    
    if not os.path.exists(wavs_dir):
        logging.error(f"Directory not found: {wavs_dir}")
        return
    
    logging.info(f"Processing WAV files in: {wavs_dir}")
    transcribe_samples(wavs_dir)
    logging.info("\nDone! Check samples.txt for the transcriptions.")

if __name__ == "__main__":
    main() 