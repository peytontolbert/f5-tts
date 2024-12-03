import os
from pathlib import Path

def create_metadata_csv(data_dir):
    """
    Convert samples.txt to metadata.csv format
    Expected metadata.csv format: wav_file|text
    """
    data_dir = Path(data_dir)
    samples_file = data_dir / "samples.txt"
    metadata_file = data_dir / "metadata.csv"
    
    # Read samples.txt
    with open(samples_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create metadata.csv with header
    with open(metadata_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("wav_file|text\n")
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split line into filename and text
            parts = line.split('|')
            if len(parts) >= 2:
                wav_file = parts[0].strip()
                text = parts[1].strip()
                
                # Ensure the wav file exists
                wav_path = data_dir / wav_file
                if wav_path.exists():
                    f.write(f"{wav_file}|{text}\n")
                else:
                    print(f"Warning: WAV file not found: {wav_file}")

if __name__ == "__main__":
    data_dir = "voice_profiles/wavs"
    create_metadata_csv(data_dir)
    print(f"Created metadata.csv in {data_dir}") 