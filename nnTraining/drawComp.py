import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def scatterPlot(pred, labs, pred_label, labs_label, file_name):
    plt.scatter(labs, pred)
    plt.xlabel(labs_label)
    plt.ylabel(pred_label)
    plt.savefig(file_name)

def compPlot(pred, labs, file_name):
    plt.plot(labs)
    plt.plot(pred)
    plt.xlabel('cells')
    plt.ylabel('$C_s$')
    plt.legend(['Ground Truth', 'Prediction'])
    plt.savefig(file_name)

count = 0
file = "compare_525_600.dat"
labs = []
pred = []
std  = 6.930475858850229653e-02
mean = -3.169763508498575307e-04

if(os.path.exists(file)):
    f = open(file, "r")
    line = f.readline()
    l = line.split()
    pred.append(float(l[0]) * std + mean)
    labs.append(float(l[1]) * std + mean)
    count += 1
    while line:
        line = f.readline()
        l = line.split()
        if l == []:
            break
        pred.append(float(l[0]) * std + mean)
        labs.append(float(l[1]) * std + mean)
        count += 1
    
    data = {'Ground Truth': labs, 'Prediction': pred}
    df = pd.DataFrame(data=data)
    plt.clf()
    sns.displot(df, kind="hist")
    plt.xlim(-0.15,0.15)
    plt.ylim(0, 60000)
    plt.savefig("displot_testing_525_600.png")
    """
    plt.scatter(labs, pred)
    #plt.hist(labs)
    #plt.hist(pred)
    plt.xlabel('C_s')
    plt.ylabel('$\hat{C}_s$')
    #plt.legend(['Ground Truth', 'Prediction'])
    plt.savefig('compare_chart_train_scatter.png')
    """  
else:
    print("no such file ".join(file))