# 💻 Hardware Requirements for OrderQ AI Training

## Recommended Hardware

For optimal performance while training and running the OrderQ AI model, the following hardware specifications are recommended:

- **Operating System**: macOS / Linux / Windows
- **CPU**: Multi-core processor with support for AVX instructions
- **GPU**: CUDA-compatible GPU (NVIDIA) with at least 4GB VRAM or Apple's M1/M2 chip with Metal Performance Shaders (MPS)
- **Memory**: Minimum 16GB RAM
- **Storage**: SSD with at least 10GB free space

## Setting Up the Environment

1. **Setup Python Virtual Environment**:
   - Use Python 3.8 or later
   - Create a virtual environment to isolate dependencies
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
2. **Install Required Packages**:
   - Use the `requirements.txt` to install all necessary packages:
   
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure GPU Support (Optional)**:
   - For systems with NVIDIA GPUs, ensure CUDA and CuDNN are installed
   - For macOS systems with M1/M2 chips, no additional setup is needed for MPS

## Running the Training

To train the model, simply execute the training script:

```bash
python train_tokenizer.py
```

## Performance Notes

- **Training Time**: Approximately 8 minutes on Apple Silicon (MPS)
- **Tips**:
  - Ensure that the virtual environment is activated during all steps
  - Monitor system resources to prevent bottlenecks
  - Use Jupyter for interactive experimentation

By following these hardware and setup guidelines, you'll be able to successfully run and train the OrderQ AI model.
