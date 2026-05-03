# AudioML
Neural Networks class repo for Audio ML Spectrogram project

# Mel Spec Model Training

Supports the following models:

- `MelTransformerFrame`
- `MelTransformerFrameBin`
- `MelTransformerFrameDelta`

The training pipeline:
1. Loads mel spectrogram datasets in ./data/raw_parquets/* and converts them to mel spectrograms (stored as .npy) with preprocess_wav_to_mel_spec.py. Must be run before training the models. 
2. Trains one of the three models based on the parameters specified in the config.json file in specmodel/
3. Evaluates predictions on validation samples (same as n_examples)
4. Generates spectrogram comparison plots in ./outputs/[ModelName]/*.png
5. Optionally reconstructs waveforms from predicted mels

---

# File Overview

Main training script example in specmodel/ with args:
Training one of the three models can be done by running the mel_trainer.py script in the specmodel/ folder after the prerequisite preprocess_wav_to_mel_spec.py script has been run. There are numerous arguments available for mel_trainer to customize training behavior, and control the resulting output. 
```bash
python mel_trainer.py --model MelTransformerFrame --epochs 30 --save_model True --n_examples 300 --patience 7
 
