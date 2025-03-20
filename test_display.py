#!/usr/bin/env python
"""
Test script to verify that generator.display("12323o", winsize=(768, 768)) works with our changes.
"""

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator

# Create a generator
generator = DisulfideClassGenerator()

# Try to display the class
try:
    print("Trying to display class 12322o...")
    generator.display("12322o", winsize=(768, 768))
    print("Success!")
except Exception as e:
    print(f"Error: {str(e)}")
