from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
PARENT = str(Path(__file__).parent)


def main():

    result_nn_pd = pd.read_csv(f"{PARENT}/result_nn.csv")
    result_snn_pd = pd.read_csv(f"{PARENT}/result_snn.csv")

    frac_list = np.unique(result_nn_pd["frac"].values)
    for frac in frac_list:
        res_nn = result_nn_pd[(result_nn_pd["frac"] == frac)]
        mat_nn=confusion_matrix(res_nn["label"].values,res_nn["estimate"].values,labels=[i for i in range(10)])
        fignn=plt.figure(1)
        imnn=plt.imshow(mat_nn)
        plt.colorbar(imnn)
        plt.xlim([-1,10])
        plt.ylim([-1,10])
        plt.xticks([i for i in range(10)])
        plt.yticks([i for i in range(10)])
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")
        plt.title(f"Confusion matrix : ratio={frac:.2f} (NN)")
        plt.savefig(f"{PARENT}/figs/nn_frac{frac}.png")
        plt.close()
        
        res_snn = result_snn_pd[(result_snn_pd["frac"] == frac)]
        mat_snn=confusion_matrix(res_snn["label"].values,res_snn["estimate"].values,labels=[i for i in range(10)])
        
        figsnn=plt.figure(2)
        imsnn=plt.imshow(mat_snn)
        plt.colorbar(imsnn)
        plt.xlim([-1,10])
        plt.ylim([-1,10])
        plt.xticks([i for i in range(10)])
        plt.yticks([i for i in range(10)])
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")
        plt.title(f"Confusion matrix : ratio={frac:.2f} (SNN)")
        plt.savefig(f"{PARENT}/figs/snn_frac{frac}.png")
        plt.close()
        
    


if __name__ == "__main__":
    main()
