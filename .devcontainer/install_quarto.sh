#!/bin/bash

set -e
set -x

# Default Quarto version
QUARTO_VERSION=${QUARTO_VER:-1.3.450}

echo "Installing Quarto version ${QUARTO_VERSION}"

# Download Quarto
wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb -O /tmp/quarto.deb

if [ $? -ne 0 ]; then
    echo "Failed to download Quarto"
    exit 1
fi

# Install Quarto
sudo dpkg -i /tmp/quarto.deb

if [ $? -ne 0 ]; then
    echo "Failed to install Quarto"
    exit 1
fi

# Clean up
rm /tmp/quarto.deb

# Verify installation
quarto check

if [ $? -ne 0 ]; then
    echo "Quarto installation check failed"
    exit 1
fi

echo "Quarto installation completed successfully"