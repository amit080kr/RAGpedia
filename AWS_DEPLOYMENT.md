# AWS Deployment Guide (Free Tier)

This guide outlines how to deploy the RAGpedia application on an AWS EC2 instance using the Free Tier (t2.micro or t3.micro).

## Prerequisites
- An AWS Account.
- Terminal (Mac/Linux) or PowerShell/PuTTY (Windows).

## Step 1: Launch an EC2 Instance

1.  **Log in to AWS Console** and navigate to **EC2**.
2.  Click **Launch Instances**.
3.  **Name**: `RAGpedia-Server`.
4.  **OS Images**: Select **Ubuntu** (Ubuntu Server 24.04 LTS or 22.04 LTS).
5.  **Instance Type**: Select `t2.micro` (or `t3.micro` if eligible for free tier).
6.  **Key Pair**:
    - Click **Create new key pair**.
    - Name: `ragpedia-key`.
    - Type: `RSA`.
    - Format: `.pem` (for Mac/Linux) or `.ppk` (for Windows/PuTTY).
    - Download the file and keep it safe!
7.  **Network Settings**:
    - Check **Allow SSH traffic from** -> **My IP** (more secure) or Anywhere.
    - Check **Allow HTTP traffic from the internet**.
    - Check **Allow HTTPS traffic from the internet**.
8.  **Configure Storage**: standard 8GB is fine.
9.  Click **Launch Instance**.

## Step 2: Configure Security Group

1.  In EC2 Dashboard, go to **Instances**, select your instance.
2.  Click the **Security** tab, then click the **Security Group ID** (e.g., `sg-xxxx`).
3.  Click **Edit inbound rules**.
4.  Add a new rule:
    - **Type**: Custom TCP
    - **Port range**: `8501` (Streamlit default port)
    - **Source**: `0.0.0.0/0` (Anywhere)
5.  Click **Save rules**.

## Step 3: Connect to the Instance

1.  Open your terminal.
2.  Move the key file to a safe folder (e.g., `~/.ssh/`).
3.  Set permissions:
    ```bash
    chmod 400 ~/.ssh/ragpedia-key.pem
    ```
4.  Connect:
    ```bash
    ssh -i "~/.ssh/ragpedia-key.pem" ubuntu@<PUBLIC-IP-ADDRESS>
    ```
    *(Replace `<PUBLIC-IP-ADDRESS>` with the Public IPv4 address from the EC2 console)*.

## Step 4: Install Docker on EC2

Once connected to the server, run:

```bash
# Update packages
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Check docker status
sudo systemctl status docker

# Allow 'ubuntu' user to run docker (avoid sudo every time)
sudo usermod -aG docker ubuntu
```
*Logout and log back in for the group change to take effect (`exit` then `ssh ...`).*

## Step 5: Deploy the App

### Option A: Clone from GitHub (Recommended)
1.  **Generate a Personal Access Token (PAT)** on GitHub if the repo is private, or just clone if public.
    ```bash
    git clone https://github.com/amit080kr/RAGpedia_deploy.git
    cd RAGpedia_deploy
    ```

### Option B: Copy files manually (if code is local only)
On your LOCAL machine:
```bash
scp -i "~/.ssh/ragpedia-key.pem" -r . ubuntu@<PUBLIC-IP-ADDRESS>:~/app
```

## Step 6: Build and Run

1.  **Build the Docker image**:
    ```bash
    docker build -t ragpedia .
    ```
    *(This may take a few minutes)*.

2.  **Run the container**:
    ```bash
    docker run -d -p 8501:8501 --name ragpedia-app --restart always ragpedia
    ```

3.  **Access the App**:
    - Open your browser and go to `http://<PUBLIC-IP-ADDRESS>:8501`.

## Troubleshooting
- **Cannot connect?** Check EC2 Security Group rules (Port 8501 must be open).
- **App crashes?** Check logs: `docker logs ragpedia-app`.
- **Permission denied?** Ensure you ran `sudo usermod -aG docker ubuntu` and re-logged in.