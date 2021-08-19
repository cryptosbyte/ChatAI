# Set up PyTorch
mkdir -p ~/pytorch/assets
cd ~/pytorch/assets
# Install Python Environment
sudo apt-get install python3-venv
python3 -m venv pytorch
# Activate PyTorch & Install Dependecies & Deactivate
source pytorch/bin/activate
pip3 install torch
pip3 install torchvision
deactivate
# Go to AI Directory
cd ~/AI
# Re-activate PyTorch Environment
source ~/pytorch/assets/pytorch/bin/activate
# Install Project-required packages
pip3 install nltk pytorch numpy
clear
# Setup NLTK & Download Punkt
python3 -c "import nltk; nltk.download('punkt');"
# Setup AI's brains | pth file
python3 src/train.py
clear
# Run AI
python3 src/chat.py