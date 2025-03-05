#!/bin/bash
# Script to install Microsoft TrueType Core Fonts on Ubuntu
# This will fix font rendering issues in proteusPy DisulfideVisualization

echo "Installing Microsoft TrueType Core Fonts..."
echo "This will require sudo privileges."

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Microsoft TrueType Core Fonts
echo "Installing ttf-mscorefonts-installer package..."
echo "Note: You will need to accept the license agreement during installation."
sudo apt-get install -y ttf-mscorefonts-installer

# Update font cache
echo "Updating font cache..."
sudo fc-cache -f -v

echo "Installation complete!"
echo "You can now run the test script to verify the fix:"
echo "python tests/test_font_fallback.py"

# Check if Arial font is now available
if [ -f "/usr/share/fonts/truetype/msttcorefonts/arial.ttf" ]; then
    echo "Success! Arial font was installed."
else
    echo "Warning: Arial font was not found in the expected location."
    echo "The font fallback mechanism in proteusPy should still work, but"
    echo "you may want to check if the installation completed successfully."
fi
