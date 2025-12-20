# Fix for Windows DLL Error with PyTorch

## Problem
If you encounter the error:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. 
Error loading "c10.dll" or one of its dependencies.
```

## Solution

This error typically occurs due to:
1. PyTorch version incompatibility with Windows
2. NumPy version incompatibility
3. Missing Visual C++ Redistributables

### Steps to Fix:

1. **Activate your virtual environment properly:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Uninstall existing PyTorch:**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```

3. **Install compatible PyTorch version (CPU):**
   ```powershell
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Ensure NumPy is compatible:**
   ```powershell
   pip install "numpy<2.0"
   ```

5. **Install Lightning:**
   ```powershell
   pip install lightning
   ```

6. **Verify installation:**
   ```powershell
   python -c "import torch; import lightning as L; print('Success!')"
   ```

## Alternative: Use Conda (if available)

If pip installation continues to fail, you can use conda:

```powershell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install lightning
```

## Important Notes

- Always activate your virtual environment before running training
- Use the venv Python explicitly: `.\.venv\Scripts\python.exe train.py ...`
- If issues persist, ensure Visual C++ Redistributables are installed from Microsoft

