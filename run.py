import mne
import matplotlib.pyplot as plt
import numpy as np
from typing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import math

# Calculating curve_lengths for 1 second of data
def calculate_curve_lengths(data):
    data = data.T
    curve_lengths = []

    for n in range(0, len(data)):
        curve_length = 0
        for k in range(0, len(data[0]) - 1):
            # curve_length += ((data[n][k + 1] - data[n][k]) ** 2 + (1/256) ** 2) ** 0.5
            curve_length += ((data[n][k + 1] - data[n][k]) ** 2) ** 0.5

        curve_lengths.append(curve_length)

    return curve_lengths

# function for calculating dmd
def dmd(X, Y):
    U2,Sig2,Vh2 = np.linalg.svd(X, False) # SVD of input matrix
    r = len(Sig2)
    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.pinv(Sig)) # build A tilde
    mu,W = np.linalg.eig(Atil)
    Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.pinv(Sig)), W) # build DMD modes
    return mu, Phi

# Calculating dmdMode_powers for 1 second of data
def calculate_dmdMode_powers(data):
    data = data.T
    dmdModePowers = []

    for i in range(0, 128, 4):
        X = data[:, i:(i + 4)]
        Y = data[:, i + 1:(i + 5)]
        mu, Phi = dmd(X, Y)
        temp = 0
        for j in range(0, len(Phi[0])):
            temp += np.linalg.norm(Phi[:, j]) ** 2
        dmdModePowers.append(temp)
    
    dmdModePowers /= sum(dmdModePowers)

    return dmdModePowers

# This function basically creates the feature datasets which we can use directly instead of wasting time creating them over and over again
def createDatasets():
    file = "dataset/chb01_"
    seizure_times = [[-1, -1], [-1, -1], [2996, 3036], [1467, 1494], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [1732, 1772], [1015, 1066], [-1, -1], [1720, 1810], [-1, -1], [-1, -1], [327, 420], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [1862, 1963], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]] 
    data_feature1 = [] # DMD Powers
    data_feature2 = [] # Curve Lengths
    labels = []
    
    # Iterating over all 42 files.
    for i in range(1, 43):
        file_data = []
        if(i < 10):
            file_data = mne.io.read_raw_edf(file + "0" + str(i) + ".edf", verbose='Error')
        else:
            file_data = mne.io.read_raw_edf(file + str(i) + ".edf", verbose='Error')
        temp_data = file_data.get_data()
        temp_data = temp_data.T
        temp_data = np.array(temp_data).reshape(int(len(temp_data) / 256), 256, 23)
        temp_data = temp_data[:, :, 0:18]

        for j in range (0, len(temp_data)):
            if(j >= seizure_times[i - 1][0] and j <= seizure_times[i - 1][1]):
                labels.append(1)
            else:
                labels.append(0)
            
            data_feature2.append(calculate_curve_lengths(temp_data[j]))
            data_feature1.append(calculate_dmdMode_powers(temp_data[j]))
        
        print("file", i, "done!")

    labels = np.array(labels)
    data_feature1 = np.array(data_feature1)
    data_feature2 = np.array(data_feature2)

    # Saving all the created arrays to files, so that we can use them directly later.
    np.savetxt("created_datasets/dmd_mode_powers.txt", data_feature1)
    np.savetxt("created_datasets/curve_lengths.txt", data_feature2)
    np.savetxt("created_datasets/labels.txt", labels)

# This is the smoothing filter if there are more than 10 consecutive 1's it keeps them as is, otherwise turns them all to zeroes
def smoothingFilter(y_pred):
    count = 0
    i = 0
    start_i = 0
    while (i < len(y_pred)):
        if(y_pred[i] == 1):
            if(count == 0):
                start_i = i
            count += 1
        else:
            if(count < 10 and count != 0):
                for j in range(start_i, i):
                    y_pred[j] = 0
            count = 0
        i += 1
    
    return 

def main():
    # Creating curve lengths and dmd mode powers
    # createDatasets()

    # Loading curve lengths and dmd mode powers from created files
    data_1 = np.loadtxt("created_datasets/dmd_mode_powers.txt") # DMD mode powers (X)
    data_2 = np.loadtxt("created_datasets/curve_lengths.txt") # Curve lengths (X)
    data = np.concatenate((data_1, data_2), axis=1)
    labels = np.loadtxt("created_datasets/labels.txt") # Y

    # Create training data and test data as given in paper (equal sized seizure seconds in both training and testing)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.5, random_state=42, stratify=labels)

    # Boosting (Default classifier is decision tree)
    clf = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R', random_state=0)
    clf.fit(X_train, Y_train) 

    # Test the performance of the model using the test data
    Y_pred = clf.predict(X_test)
    # smoothingFilter(Y_pred)

    # Printing scores.
    print("Macro F1 Score:", f1_score(Y_test, Y_pred, average='macro'))
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred))
    print("Recall:", recall_score(Y_test, Y_pred))

if __name__=='__main__':
    main()  