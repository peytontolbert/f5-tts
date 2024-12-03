import torch
import torchaudio
import sounddevice as sd
import os
import soundfile as sf
import tempfile
from types import SimpleNamespace
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    target_sample_rate,
)
import numpy as np
from pydub import AudioSegment
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from f5_tts.model.modules import MelSpec

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
model_dir = "D:/F5-TTS"
checkpoint_path = os.path.join(model_dir, "checkpoints/final_finetuned_model.pt")
vocab_path = os.path.join(model_dir, "F5TTS_Base_vocab.txt")

# Voice profile directory
VOICE_PROFILE_DIR = os.path.join(model_dir, "voice_profiles")
os.makedirs(VOICE_PROFILE_DIR, exist_ok=True)

# Load vocabulary
def load_vocab(vocab_file):
    """Load vocabulary from file and create char map"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    
    vocab_char_map = {}
    for i, char in enumerate(vocab):
        if char:  # Skip empty lines
            vocab_char_map[char] = i
            
    return vocab_char_map, len(vocab_char_map) + 1

# Initialize model
vocab_char_map, vocab_size = load_vocab(vocab_path)

model_cfg = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4
)

mel_spec_kwargs = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    mel_spec_type="vocos"
)

# Create CFM model instead of DiT directly
model = CFM(
    transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map,
)

# Load model and vocoder
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
vocoder = load_vocoder(vocoder_name="vocos", is_local=False)

def record_audio(duration=5, sample_rate=target_sample_rate):
    """Record audio from microphone and save to temporary file"""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        # Save audio to temporary file
        sf.write(temp_file.name, audio, sample_rate)
        return temp_file.name
    except Exception as e:
        os.unlink(temp_file.name)
        raise e

def get_reference_audio(prompt_text, duration=10):
    """Record reference audio for voice cloning"""
    print("\nPlease read the following text clearly and naturally:")
    print(prompt_text)
    print("(This recording will be used to improve your voice profile)")
    input("\nPress Enter when ready to record...")
    return record_audio(duration=duration)

class VoiceProfile:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.profile_dir = os.path.join(VOICE_PROFILE_DIR, profile_name)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.samples = []
        self.load_profile()

    def load_profile(self):
        """Load existing voice samples"""
        if os.path.exists(os.path.join(self.profile_dir, "samples.txt")):
            with open(os.path.join(self.profile_dir, "samples.txt"), "r") as f:
                for line in f:
                    audio_file, text = line.strip().split("|")
                    if os.path.exists(audio_file):
                        self.samples.append((audio_file, text))

    def add_sample(self, audio_file, text):
        """Add new voice sample"""
        # Copy audio file to profile directory
        new_audio_path = os.path.join(self.profile_dir, f"sample_{len(self.samples)}.wav")
        AudioSegment.from_wav(audio_file).export(new_audio_path, format="wav")
        
        # Add to samples list
        self.samples.append((new_audio_path, text))
        
        # Save samples list
        with open(os.path.join(self.profile_dir, "samples.txt"), "w") as f:
            for audio, txt in self.samples:
                f.write(f"{audio}|{txt}\n")

    def get_combined_reference(self):
        """Combine all samples into a single reference"""
        if not self.samples:
            return None, ""
        
        combined_audio = AudioSegment.empty()
        combined_text = ""
        
        for audio_file, text in self.samples:
            audio_seg = AudioSegment.from_wav(audio_file)
            combined_audio += audio_seg + AudioSegment.silent(duration=500)  # Add 0.5s silence between samples
            combined_text += text + " "
        
        # Save combined audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        combined_audio.export(temp_file.name, format="wav")
        
        return temp_file.name, combined_text.strip()

class VoiceSamplesDataset(Dataset):
    def __init__(self, voice_profile):
        self.samples = voice_profile.samples
        self.mel_spec = MelSpec(
            n_fft=1024,
            hop_length=256, 
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]
        audio, sr = torchaudio.load(audio_path)
        mel = self.mel_spec(audio)
        return {
            'text': text,
            'mel': mel,
            'audio': audio
        }

def fine_tune_model(model, profile, num_epochs=5, learning_rate=1e-5):
    """Fine-tune the model on user's voice samples"""
    print("\nFine-tuning model on your voice samples...")
    
    # Create dataset and dataloader
    dataset = VoiceSamplesDataset(profile)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Save original model state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    try:
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                output = model(
                    text=batch['text'],
                    cond=batch['audio'].to(device),
                    target=batch['mel'].to(device)
                )
                
                # Compute loss
                loss = output.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save fine-tuned model
        fine_tuned_path = os.path.join(profile.profile_dir, "fine_tuned_model.pt")
        torch.save(model.state_dict(), fine_tuned_path)
        print(f"\nFine-tuned model saved to: {fine_tuned_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError during fine-tuning: {str(e)}")
        # Restore original model state
        model.load_state_dict(original_state)
        return False

def main():
    print("F5-TTS Voice Cloning System (Continuous Learning)")
    print("---------------------------------------------")
    
    profile_name = input("Enter your voice profile name: ")
    profile = VoiceProfile(profile_name)
    
    while True:
        print("\nOptions:")
        print("1. Add new voice sample")
        print("2. Generate speech")
        print("3. Fine-tune model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            # Add new voice sample
            prompts = [
                "Hi, I'm recording this sample to create a digital copy of my voice. I want it to sound natural and conversational, just like how I normally speak.",
                "I enjoy having conversations with friends and family. We talk about our day, share stories, and discuss interesting topics that come up.",
                "When I explain things to others, I try to be clear and engaging. I use a comfortable pace and natural tone, just like I'm doing right now.",
                "Let me tell you about my favorite activities. I like to spend time outdoors, read interesting books, and learn new things whenever I can.",
                "Technology has always fascinated me. The way we can use computers and AI to create new possibilities is really amazing to think about."
            ]
            prompt = prompts[len(profile.samples) % len(prompts)]
            
            ref_audio_path = get_reference_audio(prompt)
            profile.add_sample(ref_audio_path, prompt)
            os.unlink(ref_audio_path)
            
            print("\nVoice sample added successfully!")
            
        elif choice == "2":
            if not profile.samples:
                print("\nNo voice samples recorded yet. Please add at least one sample.")
                continue
                
            # Get combined reference
            ref_audio_path, ref_text = profile.get_combined_reference()
            
            try:
                # Preprocess reference audio and text
                ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
                os.unlink(ref_audio_path)
                
                while True:
                    # Get text input
                    text = input("\nEnter text to synthesize (or 'q' to return): ")
                    if text.lower() == 'q':
                        break
                    
                    print("\nGenerating speech...")
                    # Generate speech using F5-TTS utilities
                    audio, sample_rate, _ = infer_process(
                        ref_audio,
                        ref_text,
                        text,
                        model,
                        vocoder,
                        mel_spec_type="vocos",
                        speed=1.0,
                        nfe_step=32,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1.0
                    )
                    
                    # Play synthesized audio
                    sd.play(audio, samplerate=sample_rate)
                    sd.wait()
                    
            except Exception as e:
                print(f"Error during synthesis: {str(e)}")
                
        elif choice == "3":
            if len(profile.samples) < 3:
                print("\nNeed at least 3 voice samples for fine-tuning. Please add more samples.")
                continue
            
            print("\nWarning: Fine-tuning may take several minutes and requires significant computational resources.")
            confirm = input("Do you want to proceed? (y/n): ")
            
            if confirm.lower() == 'y':
                success = fine_tune_model(model, profile)
                if success:
                    print("\nModel fine-tuning completed successfully!")
                    print("The model should now be better adapted to your voice.")
                else:
                    print("\nFine-tuning was not successful. Using original model.")
            
        elif choice == "4":
            break

if __name__ == "__main__":
    main()
