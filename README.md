# F5-TTS Voice Cloning Scripts

A collection of scripts for voice cloning using the F5-TTS model. These scripts allow you to record voice samples, train custom voice models, and generate speech with cloned voices.

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)
- F5-TTS base model and vocabulary files

Required Python packages:
```bash
pip install torch torchaudio sounddevice soundfile pydub numpy tqdm ema-pytorch
```

## Project Structure
```filestructure
├── main.py # Main script for voice recording and inference
├── train.py # Training script for fine-tuning models
├── eval.py # Evaluation and testing script
├── F5TTS_Base_vocab.txt # Base vocabulary file
└── voice_profiles/ # Directory for storing voice profiles
└── samples.txt # Voice sample metadata
```


## Setup

1. Download the F5-TTS base model and place it in the following structure:
```filestructure
D:/F5-TTS/
├── checkpoints/
│ └── final_finetuned_model.pt
└── F5TTS_Base_vocab.txt
```

2. Create a voice_profiles directory:
```bash
mkdir voice_profiles
```


## Usage

### Recording Voice Samples

Run `main.py` to start the voice recording interface:
```bash
python main.py
```


Follow the prompts to:
1. Enter your voice profile name
2. Record voice samples
3. Generate speech with your voice
4. Fine-tune the model

### Training Custom Models

To train a custom model with your voice samples:
```bash
python train.py
```


The script will:
- Load your voice samples
- Fine-tune the base model
- Save checkpoints during training
- Create a final fine-tuned model

### Evaluation

To evaluate the model or test voice cloning:
```bash
python eval.py
```


## Features

- **Voice Profile Management**: Create and manage multiple voice profiles
- **Continuous Learning**: Add new voice samples and fine-tune models
- **Real-time Synthesis**: Generate speech using your cloned voice
- **EMA Support**: Exponential Moving Average for stable training
- **Checkpoint System**: Save and load model states

## Model Configuration

Default settings:
- Sampling rate: 24kHz
- Mel channels: 100
- Hop length: 256
- Window length: 1024
- FFT size: 1024

Model architecture:
- Dimension: 1024
- Depth: 22
- Heads: 16
- Text dimension: 512
- Convolution layers: 4

## Voice Profile Format

Voice profiles are stored in the following format:
```filestructure
voice_profiles/
└── [profile_name]/
├── samples.txt # Format: audio_path|transcript
├── sample_0.wav
├── sample_1.wav
└── fine_tuned_model.pt # Profile-specific model
```



## License

[Add your license information here]

## Acknowledgments

- F5-TTS base model and implementation
- [Add other acknowledgments]