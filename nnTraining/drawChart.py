import matplotlib.pyplot as plt
import os

count = 600
file = "_train_0510235.log"
epochs = []
tloss = []
vloss = []

if(os.path.exists(file)):
    f = open(file, "r")
    f.readline()

    for e in range(count):
        line = f.readline()
        l = line.split()
        epochs.append(float(l[0]))
        tloss.append(float(l[1])*100.0)
        vloss.append(float(l[2])*100.0)
    plt.plot(epochs, tloss)
    plt.plot(epochs, vloss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss [%]')
    plt.legend(['Training', 'Validation'])
    plt.savefig('loss_chart_0510235.png')
else:
    print("no such file ".join(file))