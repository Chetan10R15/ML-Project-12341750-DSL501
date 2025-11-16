#!/bin/bash

# Quickstart Script for Feedback-Sensitive PAL
# Student: Rathod Chetankumar A (12341750)

echo "======================================================================"
echo "Feedback-Sensitive Persona-Aware ESC Chatbot - QUICKSTART"
echo "Student: Rathod Chetankumar A (12341750)"
echo "======================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install torch transformers pandas numpy matplotlib seaborn tqdm
echo "✓ Dependencies installed"
echo ""

# Train models
echo "[2/4] Training all models (this takes 3-5 minutes)..."
python3 main.py
echo "✓ Models trained"
echo ""

# Generate visualizations
echo "[3/4] Generating visualizations..."
python3 visualizations.py
echo "✓ Visualizations generated"
echo ""

# Summary
echo "[4/4] Setup complete!"
echo ""
echo "======================================================================"
echo "✅ ALL DONE! Your project is ready."
echo "======================================================================"
echo ""
echo "What was created:"
echo "  📁 models/           - 3 trained models"
echo "  📁 results/plots/    - 4 comparison plots"
echo "  📄 results/          - Evaluation results"
echo ""
echo "Next steps:"
echo "  1. View plots:    open results/plots/"
echo "  2. Read results:  cat results/summary_report.txt"
echo "  3. Try baseline:  python3 demo.py --baseline"
echo "  4. Try yours:     python3 demo.py --feedback"
echo ""
echo "For presentation:"
echo "  - Show: results/plots/1_feedback_metrics.png"
echo "  - Demo both models side by side"
echo "  - Highlight: FUR 0.0 → 0.75 (NEW CAPABILITY)"
echo ""
echo "======================================================================"
echo "Good luck with your project! 🎉"
echo "======================================================================"