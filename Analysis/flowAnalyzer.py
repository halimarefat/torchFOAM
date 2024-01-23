import numpy as np
import matplotlib.pyplot as plt
import re

def extract_coefficients(openfoam_output):
    # Regular expressions for each coefficient
    patterns = {
        'Cd': r"Cd\s*=\s*([-\d.eE]+)",
        'Cm': r"Cm\s*=\s*([-\d.eE]+)",
        'Cl': r"Cl\s*=\s*([-\d.eE]+)"
    }

    # Dictionary to store the extracted values
    extracted_values = {key: [] for key in patterns}

    # Extract and store the values for each coefficient
    for key, pattern in patterns.items():
        matches = re.findall(pattern, openfoam_output)
        extracted_values[key] = [float(value) for value in matches]

    return extracted_values

if __name__ == "__main__":
    
    file_path = '/home/hmarefat/scratch/torchFOAM/Case_NN_M4_503/casec_log.out'

    with open(file_path, 'r') as file:
        openfoam_output = file.read()

    coefficients = extract_coefficients(openfoam_output)
    #print(coefficients['Cl'][1])
    
    cd_values = coefficients['Cd'][15:]

    # Calculate the mean of Cd values
    mean_cd = np.mean(cd_values)

    # Plotting
    #plt.figure(figsize=(8, 6))
    plt.plot(cd_values, label='Cd Values', color='k')  # Plot individual Cd values with markers
    plt.axhline(y=mean_cd, color='r', linestyle='--', label=f'Mean Cd = {mean_cd:.2f}')  # Mean line
    plt.ylim([0.1,0.32])
    plt.xlabel('Timestep')
    plt.ylabel('Cd Value')
    plt.title('Cd Values and Their Mean')
    plt.legend()
    #plt.grid(True)
    #plt.plot(coefficients['Cl'], label='Cd Values', marker='o')  # Plot individual Cd values with markers
    #plt.axhline(y=np.mean(coefficients['Cl']), color='r', linestyle='-', label=f'Mean Cd = {np.mean(coefficients['Cl']):.2f}')  # Mean line
    plt.savefig('Case_NN_M4_503_Cd.png')

