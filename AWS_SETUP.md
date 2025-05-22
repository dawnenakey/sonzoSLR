# AWS Setup for OAK Camera and Annotation Platform

## Prerequisites
1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. SSH key pair for EC2 access

## Step 1: Launch EC2 Instance
1. Go to AWS Console > EC2
2. Launch Instance with these specifications:
   - AMI: Ubuntu Server 20.04 LTS
   - Instance Type: g4dn.xlarge (or similar GPU instance)
   - Storage: 30GB minimum
   - Security Group: Allow SSH (port 22) and HTTP (port 8080)
   - Key Pair: Use your existing key pair

## Step 2: Connect to Instance
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

## Step 3: Run Setup Script
```bash
# Download setup script
curl -O https://raw.githubusercontent.com/your-repo/aws_setup.sh

# Make it executable
chmod +x aws_setup.sh

# Run the script
./aws_setup.sh

# Reboot the system
sudo reboot
```

## Step 4: Start Services
After reboot, connect again and start the services:

1. Start OAK Camera:
```bash
cd ~/oak-camera
docker-compose up -d
```

2. Start Annotation Platform:
```bash
cd ~/annotation-platform
docker-compose up -d
```

## Step 5: Access Services
- OAK Camera: Connect via USB to the EC2 instance
- Annotation Platform: http://your-instance-ip:8080

## Troubleshooting
1. If OAK camera is not detected:
   ```bash
   lsusb  # Check if camera is listed
   sudo chmod 666 /dev/bus/usb/*/*  # Fix permissions
   ```

2. If annotation platform is not accessible:
   - Check security group settings
   - Verify docker containers are running:
     ```bash
     docker ps
     ```

## Additional Notes
- The setup includes:
  - Docker and Docker Compose
  - OAK camera dependencies
  - Label Studio for annotation
  - USB passthrough support
  - GPU support for better performance

## Support
For issues:
1. Check AWS EC2 instance status
2. Verify USB device detection
3. Check Docker container logs
4. Contact support with specific error messages 