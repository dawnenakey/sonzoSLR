# AWS Setup with Sign-Flow and OAK Camera

## Prerequisites
1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. SSH key pair for EC2 access
4. Sign-Flow repository access

## Step 1: Launch EC2 Instance
1. Go to AWS Console > EC2
2. Launch Instance with these specifications:
   - AMI: Ubuntu Server 20.04 LTS
   - Instance Type: g4dn.xlarge (or similar GPU instance)
   - Storage: 30GB minimum
   - Security Group: Allow SSH (port 22), HTTP (port 3000)
   - Key Pair: Use your existing key pair

## Step 2: Connect to Instance
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

## Step 3: Run Setup Script
```bash
# Download setup script
curl -O https://raw.githubusercontent.com/your-repo/aws_signflow_setup.sh

# Make it executable
chmod +x aws_signflow_setup.sh

# Run the script
./aws_signflow_setup.sh

# Reboot the system
sudo reboot
```

## Step 4: Start Services
After reboot, connect again and start the services:

```bash
cd ~/sign-flow
docker-compose up -d
```

## Step 5: Access Services
- Sign-Flow: http://your-instance-ip:3000
- OAK Camera: Connected via USB to the EC2 instance

## Sign-Flow Features
1. Video Annotation
   - Import videos
   - Add annotations
   - Export annotations
   - Timeline view

2. OAK Camera Integration
   - Real-time video capture
   - Depth sensing
   - Hand tracking
   - Gesture recognition

## Troubleshooting
1. If OAK camera is not detected:
   ```bash
   lsusb  # Check if camera is listed
   sudo chmod 666 /dev/bus/usb/*/*  # Fix permissions
   ```

2. If Sign-Flow is not accessible:
   - Check security group settings
   - Verify docker containers are running:
     ```bash
     docker ps
     ```

3. If Node.js issues:
   ```bash
   node -v  # Check Node.js version
   npm -v   # Check npm version
   ```

## Additional Notes
- The setup includes:
  - Node.js and npm
  - Docker and Docker Compose
  - OAK camera dependencies
  - Sign-Flow application
  - USB passthrough support
  - GPU support for better performance

## Support
For issues:
1. Check AWS EC2 instance status
2. Verify USB device detection
3. Check Docker container logs
4. Contact support with specific error messages 