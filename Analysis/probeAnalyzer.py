import numpy as np
import matplotlib.pyplot as plt
import re

def velDir(indx):
    if indx==0:
        return 'x'
    elif indx==1:
        return 'y'
    elif indx==2:
        return 'z'

def probeFileReader(path, skipped):
    _time = []  
    _probe_data = [] 
    try:
        with open(path, 'r') as file:
            for _ in range(skipped):
                next(file)
                
            for line in file:
                parts = line.strip().split()
                if parts:
                    time_value = float(parts[0])
                    _time.append(time_value)

                    vector_strings = re.findall(r"\((.*?)\)", line)
                    vectors = [tuple(map(float, vec.split())) for vec in vector_strings]
                    _probe_data.append(vectors)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return _time, _probe_data

def energySpectrum(u):
    u1 = u - np.mean(u)
    N = len(u1)
    uhat = np.fft.fft(u1)
    uhat = uhat[:N//2 + 1]
    f = np.linspace(0, np.pi, len(uhat)) 
    _NormF = (1 / np.pi) * f
    _e = 1 / (2 * np.pi * N) * np.abs(uhat)**2
    
    return _NormF, _e


if __name__ == "__main__":
        
    file_path_R103 = '/home/hmarefat/scratch/torchFOAM/Case_dS/postProcessing/probes/0/U'
    file_path_R503 = '/home/hmarefat/scratch/torchFOAM/Case_dS_R53/postProcessing/probes/0/U'
    file_path_R104 = '/home/hmarefat/scratch/torchFOAM/Case_dS_R4/postProcessing/probes/0/U'

    skpLines = 58
    time_R103, probe_data_R103 = probeFileReader(file_path_R103, skpLines)
    time_R503, probe_data_R503 = probeFileReader(file_path_R503, skpLines)
    time_R104, probe_data_R104 = probeFileReader(file_path_R104, skpLines)
    
    flag = 'partial' #'full' #
    file_path = file_path_R104
    probes = np.array(probe_data_R104)
    #print(probes.shape)

    
    if flag == 'partial':
        probes_R103 = np.array(probe_data_R103)
        probes_R503 = np.array(probe_data_R503)
        probes_R104 = np.array(probe_data_R104)
        probe_indices = [1, 2, 3, 5, 6, 15, 16, 17, 19, 20, 36, 37, 38, 40, 41, 50, 51, 52, 54, 55]
        dir_index = 1
        for probe_index in probe_indices:
            NormF_R103, e_R103 = energySpectrum(probes_R103[:, probe_index, dir_index])
            NormF_R503, e_R503 = energySpectrum(probes_R503[:, probe_index, dir_index])
            NormF_R104, e_R104 = energySpectrum(probes_R104[:, probe_index, dir_index])
            plt.figure(figsize=(10, 6))
            plt.loglog(NormF_R103, np.convolve(e_R103, np.ones(20)/20, mode='same'), '-b', label=r'$Re = 10^3 $')
            plt.loglog(NormF_R503, np.convolve(e_R503, np.ones(20)/20, mode='same'), '-r', label=r'$Re = 5\times 10^3 $')
            plt.loglog(NormF_R104, np.convolve(e_R104, np.ones(20)/20, mode='same'), '-g', label=r'$Re = 10^4 $')
            plt.loglog(NormF_R103[1:], 1e-2 * NormF_R103[1:]**(-5/3), '--k', label='-5/3 Slope')
            plt.xlabel("$f$", fontsize=17)
            plt.ylabel("$E_{vv}/{U^2D}$", fontsize=17)
            plt.grid(True, which="both", ls="-")
            #plt.text(0.1, 1, '-5/3', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend()

                    # Save the plot
            save_path = f'./New_Compare_energy_spectrum_probe_{probe_index}_U{velDir(dir_index)}.png'  # Update with your desired save path
            plt.savefig(save_path)
            plt.close()
        
    elif flag == 'full':
        probe_count = probes.shape[1]
        for probe_index in range(probe_count):
            for dir_index in range(3):
                NormF, e = energySpectrum(probes[:, probe_index, dir_index])

                # Plotting
                plt.figure(figsize=(10, 6))
                plt.loglog(NormF, np.convolve(e, np.ones(5)/5, mode='same'), '-k', label='Energy Spectrum')
                plt.loglog(NormF[1:], 1e-2 * NormF[1:]**(-5/3), '--k', label='-5/3 Slope')
                plt.xlabel("$f$", fontsize=17)
                plt.ylabel("$E_{uu}/{U^2D}$", fontsize=17)
                plt.grid(True, which="both", ls="-")
                #plt.text(0.1, 1, '-5/3', fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend()

                # Save the plot
                save_path = f'./dS_Plots/{file_path.split("/")[-5]}_energy_spectrum_probe_{probe_index}_U{velDir(dir_index)}.png'  # Update with your desired save path
                plt.savefig(save_path)
                plt.close()
                
                print(f'The plot for probe {probe_index} of U{velDir(dir_index)} is saved!')
