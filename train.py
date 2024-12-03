import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import json
from f5_tts.model import CFM, DiT
from f5_tts.model.modules import MelSpec
import torch.optim as optim
from tqdm import tqdm
from ema_pytorch import EMA
from pathlib import Path
from f5_tts.model.utils import convert_char_to_pinyin

# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"

# Training settings
learning_rate = 1e-5
batch_size = 4
epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vocab(vocab_file):
    """Load vocabulary from file and create char map"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    
    # Create vocabulary mapping
    vocab_char_map = {}
    for i, char in enumerate(vocab):
        if char:  # Skip empty lines
            vocab_char_map[char] = i
            
    return vocab_char_map, len(vocab_char_map) + 1  # Add 1 for padding token

class WavsDataset(Dataset):
    def __init__(self, data_dir, mel_spec_kwargs):
        self.data_dir = Path(data_dir)
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        
        # Try loading metadata.csv first, fall back to samples.txt if needed
        metadata_path = self.data_dir / "metadata.csv"
        samples_path = self.data_dir / "samples.txt"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                # Skip header
                next(f)
                self.samples = [line.strip().split('|') for line in f if line.strip()]
        elif samples_path.exists():
            with open(samples_path, 'r', encoding='utf-8') as f:
                self.samples = [line.strip().split('|') for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Neither metadata.csv nor samples.txt found in {self.data_dir}")
            
        # Load durations if available
        duration_path = self.data_dir / "duration.json"
        if duration_path.exists():
            with open(duration_path, 'r') as f:
                self.durations = json.load(f)['duration']
        else:
            self.durations = None
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wav_path, text = self.samples[idx]
        wav_path = self.data_dir / wav_path
        
        # Load audio
        audio, sr = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Generate mel spectrogram
        mel = self.mel_spec(audio)  # Shape: [1, channels, time]
        
        # Verify mel dimensions and reshape if necessary
        if mel.dim() == 2:  # If mel is [channels, time]
            mel = mel.unsqueeze(0)  # Add batch dimension to make [1, channels, time]
        elif mel.dim() == 3 and mel.shape[1] == 1:  # If mel is [1, 1, time]
            mel = mel.transpose(1, 2)  # Make it [1, time, 1]
            mel = mel.expand(-1, -1, n_mel_channels)  # Expand to [1, time, n_mel_channels]
            mel = mel.transpose(1, 2)  # Back to [1, n_mel_channels, time]
        
        # Final verification
        assert mel.shape[1] == n_mel_channels, f"Expected {n_mel_channels} mel channels, got {mel.shape[1]}"
        
        return {
            'audio': audio,
            'mel': mel,
            'text': text,
            'duration': self.durations[idx] if self.durations else None,
            'mel_lengths': mel.shape[-1]  # Length is the last dimension
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Get max length in the batch
    max_mel_len = max(item['mel'].shape[-1] for item in batch)
    
    # Prepare lists for batch items
    audio_batch = []
    mel_batch = []
    text_batch = []
    mel_lengths = []
    durations = []
    
    # Pad sequences to max length
    for item in batch:
        mel = item['mel']
        curr_len = mel.shape[-1]
        
        # Calculate padding
        pad_len = max_mel_len - curr_len
        
        # Pad mel spectrogram
        padded_mel = torch.nn.functional.pad(mel, (0, pad_len), mode='constant', value=0)
        
        # Store items
        mel_lengths.append(curr_len)
        mel_batch.append(padded_mel)
        audio_batch.append(item['audio'])
        text_batch.append(item['text'])
        if item['duration'] is not None:
            durations.append(item['duration'])
    
    # Stack tensors
    mel_batch = torch.stack(mel_batch, 0)
    mel_lengths = torch.tensor(mel_lengths)
    
    return {
        'audio': audio_batch,  # Keep as list since lengths vary
        'mel': mel_batch,
        'text': text_batch,
        'duration': torch.tensor(durations) if durations else None,
        'mel_lengths': mel_lengths
    }

def train_model(model, train_loader, optimizer, ema_model, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move data to device
            mel = batch['mel'].to(device)  # Shape: [batch, 1, channels, time]
            
            # Reshape mel: [batch, 1, channels, time] -> [batch, time, channels]
            mel = mel.squeeze(1)  # Remove the extra dimension
            mel = mel.transpose(1, 2)  # Swap channels and time dimensions
            
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Process text data - convert to pinyin first
            text_batch = batch['text']
            
            # Convert text to pinyin
            pinyin_texts = convert_char_to_pinyin(text_batch, polyphone=True)
            
            # Convert pinyin to tensor using vocab_char_map
            encoded_text = []
            for pinyin_chars in pinyin_texts:
                # Join the character list into a single string
                text = ''.join(pinyin_chars)
                # Convert to tensor using vocab_char_map
                encoded = torch.tensor([model.vocab_char_map.get(c, 0) for c in text], 
                                    dtype=torch.long,
                                    device=device)
                encoded_text.append(encoded)
            
            # Pad text sequences to max length in batch
            max_text_len = max(len(t) for t in encoded_text)
            padded_text = torch.zeros((len(encoded_text), max_text_len), 
                                    dtype=torch.long,
                                    device=device)
            for i, text in enumerate(encoded_text):
                padded_text[i, :len(text)] = text
            
            # Forward pass with both mel and text inputs
            loss, _, _ = model(inp=mel, text=padded_text, lens=mel_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update EMA model
            ema_model.update()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/finetuned_model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

def main():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    vocab_file = "vocab.txt"
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
    
    # Load vocabulary
    vocab_char_map, vocab_size = load_vocab(vocab_file)
    
    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    
    # Initialize model
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4
    )
    
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    
    # Create EMA model
    ema_model = EMA(model, beta=0.9999, update_after_step=100)
    
    # Load pretrained model if exists
    checkpoint_path = "checkpoints/model_1200000.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema_model_state_dict' in checkpoint:
            state_dict = checkpoint['ema_model_state_dict']
            # Remove unwanted keys and 'ema_model.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('ema_model.'):
                    key = key.replace('ema_model.', '')
                if key not in ['initted', 'step'] and not key.startswith('mel_spec.mel_stft'):
                    new_state_dict[key] = value
            
            # Load filtered state dict
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded pretrained weights")
    
    model = model.to(device)
    ema_model.to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = WavsDataset('voice_profiles/wavs', mel_spec_kwargs)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Train the model
    train_model(model, train_loader, optimizer, ema_model, device, epochs)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
    }, 'checkpoints/final_finetuned_model.pt')
    print("Training completed! Final model saved.")

if __name__ == "__main__":
    main() 