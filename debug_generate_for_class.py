#!/usr/bin/env python
"""
Debug script to investigate why generate_for_class("--+--") fails.
"""

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator

# Create a generator
generator = DisulfideClassGenerator()

# Print some debug info
print(f"Base: {generator.base}")
print(f"DataFrame shape: {generator.df.shape}")
print(f"DataFrame columns: {generator.df.columns.tolist()}")

# Check if "--+--" is in the class_str column
if "class_str" in generator.binary_df.columns:
    matching_rows = generator.df[generator.df["class_str"] == "--+--"]
    print(f"Rows matching class_str='--+--': {len(matching_rows)}")
    if not matching_rows.empty:
        print(f"First matching row: {matching_rows.iloc[0].to_dict()}")
else:
    print("No 'class_str' column in DataFrame")

# Try to parse the class string
base, clean_string = generator.parse_class_string("--+--")
print(f"Parsed class string: base={base}, clean_string={clean_string}")

# Try to generate disulfides for the class
try:
    # Try with use_class_str=True
    print("\nTrying with use_class_str=True:")
    sslist = generator.generate_for_class("--+--", use_class_str=True)
    if sslist is None:
        print("Result: None")
    else:
        print(f"Result: DisulfideList with {len(sslist)} items")
except Exception as e:
    print(f"Error: {str(e)}")

try:
    # Try with use_class_str=False
    print("\nTrying with use_class_str=False:")
    sslist = generator.generate_for_class("--+--", use_class_str=False)
    if sslist is None:
        print("Result: None")
    else:
        print(f"Result: DisulfideList with {len(sslist)} items")
except Exception as e:
    print(f"Error: {str(e)}")
