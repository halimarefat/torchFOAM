import matplotlib.pyplot as plt

count = 273
f = open("_train.log", "r")
f.readline()
epochs = []
tloss = []
vloss = []

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
plt.savefig('loss_chart.png')