#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    libusb-1.0-0-dev \
    libusb-1.0-0 \
    libudev-dev \
    git \
    awscli

# Install Python dependencies
pip3 install \
    depthai \
    boto3 \
    numpy \
    opencv-python \
    awscli

# Configure AWS CLI
aws configure

# Create project directory
mkdir -p ~/spokhand
cd ~/spokhand

# Clone repository
git clone https://github.com/spokhand/spokhandSLR.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Create systemd service for OAK camera
sudo tee /etc/systemd/system/spokhand-camera.service << 'EOL'
[Unit]
Description=Spokhand OAK Camera Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/spokhand
Environment=PATH=/home/$USER/spokhand/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/$USER/spokhand/venv/bin/python src/camera_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Enable and start service
sudo systemctl enable spokhand-camera
sudo systemctl start spokhand-camera

# Create data directory
mkdir -p ~/spokhand/data

# Set up AWS S3 sync
sudo tee /etc/cron.d/spokhand-sync << 'EOL'
*/5 * * * * $USER aws s3 sync /home/$USER/spokhand/data s3://spokhand-videos-dev/raspberry-pi/$(hostname)/
EOL

echo "Setup complete! The OAK camera service is now running."
echo "Data will be synced to AWS S3 every 5 minutes."
echo "To check service status: sudo systemctl status spokhand-camera"
echo "To view logs: sudo journalctl -u spokhand-camera -f" 