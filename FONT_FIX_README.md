# Font Rendering Fix for Linux

This document explains the issue with text rendering in disulfide list visualizations on Linux and provides solutions.

## The Problem

The issue occurs when displaying disulfide lists in Linux, where text titles are not visible. This happens because:

1. The code was trying to use Arial font, which is not available by default on most Linux distributions
2. When the font wasn't found, the code didn't have a proper fallback mechanism

## Solutions

### 1. Code Changes (Already Implemented)

The following changes have been made to `utility.py`:

1. Enhanced the `find_arial_font()` function to:
   - Look for common Linux system fonts (DejaVu Sans, Liberation Sans, Free Sans) if Arial is not found
   - Use `fc-list` as a last resort to find any available TTF font on the system

2. Improved the `calculate_fontsize()` function to:
   - Handle the case where no font is found by using a reasonable estimation based on title length and window width
   - Add better error handling for font loading issues

### 2. Install Microsoft TrueType Core Fonts (Recommended)

For a more permanent solution, install the Microsoft TrueType Core Fonts package:

```bash
# Update package lists
sudo apt-get update

# Install Microsoft TrueType Core Fonts
sudo apt-get install ttf-mscorefonts-installer

# Update font cache
sudo fc-cache -f -v
```

During installation, you'll need to accept the license agreement.

### 3. Testing the Fix

A test script has been created to verify that the font fallback mechanism works correctly:

```bash
# Run the test script
python tests/test_font_fallback.py
```

This script will:
1. Check if a suitable font is found
2. Display a single disulfide
3. Display a disulfide list
4. Display an overlay visualization

If text titles are visible in all visualizations, the fix was successful.

## Technical Details

The font fallback mechanism works in the following order:

1. Try to find Arial font in standard locations
2. Look for common Linux system fonts (DejaVu Sans, Liberation Sans, Free Sans)
3. Use `fc-list` to find any available TTF font
4. If no font is found, estimate an appropriate font size based on title length and window width

This ensures that text will be displayed even if no suitable font is found, though the sizing may not be optimal in that case.
