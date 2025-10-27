#!/bin/bash
# Hull Tactical Market Prediction - UV Setup Script (Fast!)

echo "============================================================"
echo "Hull Tactical Market Prediction - UV Environment Setup"
echo "============================================================"

echo ""
echo "1. Creating virtual environment with uv..."
uv venv hull_tactical_env

echo ""
echo "2. Activating virtual environment..."
source hull_tactical_env/bin/activate

echo ""
echo "3. Installing dependencies with uv (much faster!)..."
uv pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost tensorflow optuna "optuna-integration[lightgbm]" TA-Lib jupyter ipykernel notebook tqdm

echo ""
echo "4. Running setup verification..."
python test_environment.py

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment in the future:"
echo "  source hull_tactical_env/bin/activate"
echo ""
echo "To run the solution:"
echo "  jupyter notebook hull_tactical_prediction.ipynb"
echo ""
echo "🚀 UV made this setup much faster!"
