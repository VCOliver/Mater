from mater import *

# Define variables at module level
x = 10
y = [0,1,2]

if __name__ == "__main__":
    print("Variables before loading:")
    who()
    
    # Test loading (assuming you have ECG_1.mat file)
    try:
        load("test_files/ECG_1.mat")
        print("\nVariables after loading:")
        who()
        
        # Test if specific variables are accessible
        if 'fs' in globals():
            print(f"\nfs = {fs}") # type: ignore
        if 'x' in globals():
            print(f"x shape = {x.shape if hasattr(x, 'shape') else type(x)}") # type: ignore
            
    except FileNotFoundError:
        print("\nECG_1.mat not found, testing with test_data.mat")
        try:
            load("test_data.mat")
            print("\nVariables after loading test_data.mat:")
            who()
        except FileNotFoundError:
            print("No .mat files found to test with")