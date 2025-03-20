#!/usr/bin/env python
"""
Debug script to investigate why generator.display("12323o", winsize=(768, 768)) fails.
"""

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator
import pandas as pd

# Create a generator
generator = DisulfideClassGenerator()

# Print some debug info
print(f"Base: {generator.base}")
print(f"Binary DataFrame shape: {generator.binary_df.shape if generator.binary_df is not None else None}")
print(f"Octant DataFrame shape: {generator.octant_df.shape if generator.octant_df is not None else None}")

# Parse the class string
class_id = "12323o"
base, clean_string = generator.parse_class_string(class_id)
print(f"Parsed class string: base={base}, clean_string={clean_string}")

# Check if the class is in the dictionaries
print(f"Class in binary_class_disulfides: {clean_string in generator.binary_class_disulfides}")
print(f"Class in octant_class_disulfides: {clean_string in generator.octant_class_disulfides}")

# Check if the class is in the dataframes
if generator.octant_df is not None:
    print(f"Class in octant_df (class column): {clean_string in generator.octant_df['class'].values}")
    print(f"Class in octant_df (class_str column): {clean_string in generator.octant_df['class_str'].values}")
    
    # Print the data types
    print(f"Type of clean_string: {type(clean_string)}")
    print(f"Type of first value in class column: {type(generator.octant_df['class'].values[0])}")
    
    # Print the first few values in the class column
    print(f"First 10 values in class column: {generator.octant_df['class'].values[:10]}")
    
    # Try converting to different types
    print(f"Class in octant_df (class column) as int: {int(clean_string) in generator.octant_df['class'].astype(int).values}")
    print(f"Class in octant_df (class column) as str: {clean_string in generator.octant_df['class'].astype(str).values}")
    
    # Check for similar class strings
    similar_classes = [val for val in generator.octant_df['class'].values if val.startswith(clean_string[:4])]
    print(f"Similar classes: {similar_classes[:10] if similar_classes else None}")
    
    # Print all unique class values
    unique_classes = generator.octant_df['class'].unique()
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Unique classes containing '123': {[c for c in unique_classes if '123' in c]}")

# Try to generate disulfides for the class
try:
    print("\nTrying to generate disulfides for the class...")
    disulfide_list = generator.generate_for_class(clean_string, use_class_str=False)
    if disulfide_list is None:
        print("Result: None")
    else:
        print(f"Result: DisulfideList with {len(disulfide_list)} items")
except Exception as e:
    print(f"Error: {str(e)}")

# Try with the original class_id
try:
    print("\nTrying with the original class_id...")
    disulfide_list = generator.generate_for_class(class_id)
    if disulfide_list is None:
        print("Result: None")
    else:
        print(f"Result: DisulfideList with {len(disulfide_list)} items")
except Exception as e:
    print(f"Error: {str(e)}")

# Try with a similar class that does exist
similar_class = "12322"  # From the first 10 values we saw
try:
    print(f"\nTrying with a similar class that does exist: {similar_class}...")
    disulfide_list = generator.generate_for_class(similar_class, use_class_str=False)
    if disulfide_list is None:
        print("Result: None")
    else:
        print(f"Result: DisulfideList with {len(disulfide_list)} items")
except Exception as e:
    print(f"Error: {str(e)}")
