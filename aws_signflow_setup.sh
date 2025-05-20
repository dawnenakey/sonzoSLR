#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install OAK camera dependencies
sudo apt-get install -y \
    libusb-1.0-0-dev \
    libusb-1.0-0 \
    libudev-dev \
    libusb-1.0-0-dbg

# Create directory for Sign-Flow
mkdir -p ~/sign-flow
cd ~/sign-flow

# Clone Sign-Flow repository
git clone https://github.com/your-repo/sign-flow.git .

# Install Sign-Flow dependencies
npm install

# Create docker-compose.yml for OAK camera
cat > docker-compose.yml << 'EOL'
version: '3'
services:
  oak-camera:
    image: luxonis/depthai:latest
    privileged: true
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - ./data:/data
    environment:
      - DISPLAY=${DISPLAY}
    network_mode: host

  sign-flow:
    build: .
    ports:
      - "3000:3000"
    volumes:
      - ./data:/app/data
    environment:
      - NODE_ENV=production
      - VITE_API_URL=http://localhost:3000
    depends_on:
      - oak-camera
EOL

# Create Dockerfile for Sign-Flow
cat > Dockerfile << 'EOL'
FROM node:18

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Start the application
CMD ["npm", "start"]
EOL

# Create .env file for Sign-Flow
cat > .env << 'EOL'
VITE_API_URL=http://localhost:3000
NODE_ENV=production
EOL

echo "Setup complete! Please reboot your system."
echo "After reboot:"
echo "1. Start the services: cd ~/sign-flow && docker-compose up -d"
echo "2. Access Sign-Flow at: http://your-instance-ip:3000" 