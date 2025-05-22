#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

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

# Create directory for OAK camera
mkdir -p ~/oak-camera
cd ~/oak-camera

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
EOL

# Create directory for Sign-Flow
mkdir -p ~/sign-flow
cd ~/sign-flow

# Create docker-compose.yml for Sign-Flow
cat > docker-compose.yml << 'EOL'
version: '3'
services:
  sign-flow:
    build: .
    ports:
      - "3000:3000"
    volumes:
      - ./data:/app/data
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/signflow
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=signflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
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

echo "Setup complete! Please reboot your system."
echo "After reboot:"
echo "1. For OAK camera: cd ~/oak-camera && docker-compose up -d"
echo "2. For Sign-Flow: cd ~/sign-flow && docker-compose up -d" 