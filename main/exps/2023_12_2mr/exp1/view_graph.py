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
    scores = []
    for frac in frac_list:
        res_nn = result_nn_pd[(result_nn_pd["frac"] == frac)]
        acc_nn=accuracy_score(res_nn["label"].values,res_nn["estimate"].values)
        print(f"{frac}, {acc_nn}")
        
        res_snn = result_snn_pd[(result_snn_pd["frac"] == frac)]
        acc_snn=accuracy_score(res_snn["label"].values,res_snn["estimate"].values)
        print(f"{frac}, {acc_snn}")
        print("---")
        
        scores.append([acc_nn,acc_snn])
        
    scores_pd=pd.DataFrame(scores,columns=["nn","snn"])
    plt.plot(frac_list,scores_pd["nn"],label="NN")
    plt.plot(frac_list,scores_pd["snn"],label='SNN')
    plt.grid()
    plt.legend()
    plt.xlabel("Ratio of contact force")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
