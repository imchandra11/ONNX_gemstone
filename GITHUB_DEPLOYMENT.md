# GitHub Deployment Guide

This guide will help you deploy your Gemstone Price Prediction project to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account created
- Repository created on GitHub (or we'll create one)

## Step-by-Step Deployment

### 1. Initialize Git Repository (if not already initialized)

```bash
git init
```

### 2. Add All Files to Git

```bash
git add .
```

### 3. Create Initial Commit

```bash
git commit -m "Initial commit: Gemstone price prediction project with PyTorch Lightning and FastAPI"
```

### 4. Create Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `gemstone-price-predictor`)
5. Choose public or private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 5. Add Remote Repository

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Or if using SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

### 6. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

## Complete Command Sequence

Here's the complete sequence of commands to run:

```bash
# Navigate to your project directory
cd "C:\Users\91838\Desktop\New folder"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Gemstone price prediction project with PyTorch Lightning and FastAPI"

# Add remote (replace with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Important Notes

### Large Files

- The `gemstone.csv` file (10MB) is included in the repository. If you want to exclude it:
  1. Uncomment the line `# gemstone.csv` in `.gitignore`
  2. Remove it from git: `git rm --cached gemstone.csv`
  3. Commit the change: `git commit -m "Remove large CSV file from repository"`

### Model Files

- Model files in the `models/` directory are excluded by default (they're generated files)
- Users will need to train the model themselves using `train.py`

### Virtual Environment

- The `.venv/` directory is excluded (users should create their own)

## Future Updates

To push future changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Cloning the Repository

Others can clone your repository using:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

Then they should:
1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train.py --csv gemstone.csv`
4. Run the app: `uvicorn app.main:app --reload`

## Troubleshooting

### If you get "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### If you need to update remote URL
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### If push is rejected
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Repository Structure on GitHub

Your repository will contain:
- ✅ Source code (`train.py`, `app/`)
- ✅ Configuration files (`requirements.txt`, `README.md`)
- ✅ Dataset (`gemstone.csv` - if not excluded)
- ❌ Virtual environment (excluded)
- ❌ Model files (excluded - users train their own)
- ❌ Logs and cache files (excluded)

