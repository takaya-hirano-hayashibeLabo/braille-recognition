from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
PARENT = str(Path(__file__).parent)


def main():

    result_nn_pd = pd.read_csv (f"{PARENT}/sloped_result_nn.csv")
    result_snn_pd = pd.read_csv(f"{PARENT}/sloped_result_snn.csv")


    res_nn = result_nn_pd
    mat_nn=confusion_matrix(res_nn["label"].values,res_nn["estimate"].values,labels=[i for i in range(10)])
    print(classification_report(res_nn["label"].values,res_nn["estimate"].values,labels=[i for i in range(10)]))
    fignn=plt.figure(1)
    imnn=plt.imshow(mat_nn)
    plt.colorbar(imnn)
    plt.xlim([-1,10])
    plt.ylim([-1,10])
    plt.xticks([i for i in range(10)])
    plt.yticks([i for i in range(10)])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Confusion matrix (NN)")
    plt.savefig(f"{PARENT}/figs/nn.png")
    plt.close()
    
    res_snn = result_snn_pd
    mat_snn=confusion_matrix(res_snn["label"].values,res_snn["estimate"].values,labels=[i for i in range(10)])
    print(classification_report(res_snn["label"].values,res_snn["estimate"].values,labels=[i for i in range(10)]))
    
    figsnn=plt.figure(2)
    imsnn=plt.imshow(mat_snn)
    plt.colorbar(imsnn)
    plt.xlim([-1,10])
    plt.ylim([-1,10])
    plt.xticks([i for i in range(10)])
    plt.yticks([i for i in range(10)])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Confusion matrix (SNN)")
    plt.savefig(f"{PARENT}/figs/snn.png")
    plt.close()
        
    


if __name__ == "__main__":
    main()
