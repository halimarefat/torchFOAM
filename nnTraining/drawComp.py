import matplotlib.pyplot as plt
import os

count = 0
file = "compare_training.dat"
labs = []
pred = []

if(os.path.exists(file)):
    f = open(file, "r")
    line = f.readline()
    l = line.split()
    pred.append(float(l[0]))
    labs.append(float(l[1]))
    count += 1
    while line:
        line = f.readline()
        l = line.split()
        if l == []:
            break
        pred.append(float(l[0]))
        labs.append(float(l[1]))
        count += 1
        

    plt.scatter(labs, pred)
    #plt.hist(labs)
    #plt.hist(pred)
    plt.xlabel('C_s')
    plt.ylabel('$\hat{C}_s$')
    #plt.legend(['Ground Truth', 'Prediction'])
    plt.savefig('compare_chart_train_scatter.png')
else:
    print("no such file ".join(file))