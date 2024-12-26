# %%
import random, pywt, statistics
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# %%
FS = 360
LOWCUT = 0.5
HIGHCUT = 40
ORDER = 4
WAVELET_LEVEL = 2
WAVELET_DB = 'db3'

# %%
LBBB_test_path = './Data/Normal&LBBB/LBBB_Test.txt'
LBBB_train_path = './Data/Normal&LBBB/LBBB_Train.txt'
normal_test_path = './Data/Normal&LBBB/Normal_Test.txt'
normal_train_path = './Data/Normal&LBBB/Normal_Train.txt'

# %%
def getDataEntries(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split('|')
            data.append(parts)
    data = [i[:len(i)-1] for i in data]
    new_data = []
    for inner_list in data:
        new_inner_list = [float(element) for element in inner_list]
        new_data.append(new_inner_list)
    return new_data


# %%
def plot_sample(data, label = "Signal"):
    fig, axes = plt.subplots(1, len(data), figsize=(15, 2), sharey=True)
    i = 0
    for d in data:
        t = np.linspace(0, FS, len(d))
        axes[i].plot(t, d)
        axes[i].set_title(f"{label} {i+1}")
        axes[i].grid(True)
        i+=1
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Reading Data

# %%
# reading data and constructing their labels
LBBB_test_data = getDataEntries(LBBB_test_path)
LBBB_target_test = ["LBBB" for i in range(len(LBBB_test_data))]

normal_test_data = getDataEntries(normal_test_path)
normal_target_test = ["Normal" for i in range(len(normal_test_data))]

LBBB_train_data = getDataEntries(LBBB_train_path)
LBBB_target_train = ["LBBB" for i in range(len(LBBB_train_data))]

normal_train_data = getDataEntries(normal_train_path)
normal_target_train = ["Normal" for i in range(len(normal_train_data))]

print(len(LBBB_train_data))


# %%
plot_sample(LBBB_train_data[0:4], "LBBB singal - Train")
plot_sample(normal_train_data[0:4], "Normal singal - Train")

# %% [markdown]
# # preprocessing 

# %%
def remove_duplicates(data):
    new_data = []
    seen = set()
    for row in data:
        entry = tuple(row) 
        if entry not in seen:
            seen.add(entry)
            new_data.append(list(row)) 
    return new_data

# %%
# mean removal to remove signal shifting
def mean_removal(data):
    new_data = []
    for row in data:
        processed_row = []
        mean = statistics.mean(row)
        for sample in row:
            processed_row.append(sample - mean)
        new_data.append(processed_row)
    return new_data


# %%
# signal normalization from -1 to 1 to transform features to be on a similar scale
def normalize_signal(data):
    new_data = []
    for row in data:
        data = np.array(row)
        x_min = data.min()
        x_max = data.max()
        processed_row = 2 * (data - x_min) / (x_max - x_min) - 1
        new_data.append(processed_row)
    return new_data

# %%
def butter_bandpass_filter(data):
    new_data = []
    for row in data: 
        # deviding by nyquist freq to prevent aliasing 
        nyquist = 0.5 * FS  
        low = LOWCUT / nyquist
        high = HIGHCUT / nyquist
        b, a = butter(ORDER, [low, high], btype='band')
        # apply the filter to the signal with zero phase shift
        new_data.append(filtfilt(b, a, row)) 
    return new_data

# %%
def preprocess(data):
    new_data = remove_duplicates(data)
    new_data = mean_removal(new_data)
    new_data = butter_bandpass_filter(new_data)
    new_data = normalize_signal(new_data)
    return new_data

# %% [markdown]
# ##### preprocessing train and test data

# %%
normal_train_data = preprocess(normal_train_data)
normal_target_train =  ["Normal" for i in range(len(normal_train_data))]
# print("lenth of train lbb before preprocessing",len(LBBB_train_data))
# there's around 200 sample duplicated in lbbb train data
LBBB_train_data = preprocess(LBBB_train_data)
LBBB_target_train =  ["LBBB" for i in range(len(LBBB_train_data))]
# print("lenth of train lbb after preprocessing",len(LBBB_train_data))
LBBB_test_data = preprocess(LBBB_test_data)
LBBB_target_test =  ["LBBB" for i in range(len(LBBB_test_data))]

normal_test_data = preprocess(normal_test_data)
normal_target_test =  ["Normal" for i in range(len(normal_test_data))]

# %%
plot_sample(LBBB_train_data[0:4], "LBBB processed - Train")
plot_sample(normal_train_data[0:4], "Normal processed - Train")

# %% [markdown]
# #### combine and shuffle data

# %%
print(len(LBBB_train_data))
print(len(normal_train_data))
def combine_shuffle(data1, data2, target1, target2):
    data1.extend(data2)
    target1.extend(target2)
    
    return target1, data1

# %%
test_labels, test_data = combine_shuffle(LBBB_test_data, normal_test_data, LBBB_target_test, normal_target_test)
print(LBBB_target_train)
print(normal_target_train)
train_labels, train_data = combine_shuffle(LBBB_train_data, normal_train_data, LBBB_target_train, normal_target_train)

# %% [markdown]
# # Feature Extraction

# %%
# coeffs_trainstructs -> [a, d1, d2, d3, d4, ..]
def wavedec(data):
    return pywt.wavedec(data, wavelet=WAVELET_DB, level=WAVELET_LEVEL)
coeffs_test = wavedec(test_data)
coeffs_train = wavedec(train_data)

# %%
a_train = coeffs_train[0]
details_train = coeffs_train[1:]

a_test = coeffs_test[0]
details_test = coeffs_test[1:]

# %%
def compute_statistics(values):
    # Ensure values is a numpy array for easier calculations
    values = np.array(values)

    min_val = np.min(values)
    max_val = np.max(values)
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_dev = np.std(values)
    skewness = skew(values)
    kurt = kurtosis(values)

    # min - max - mean - median - std deviation - skew - kurt
    return [min_val, max_val, mean_val, median_val, std_dev, skewness, kurt]

# %%
train_data = [compute_statistics(a_train[i]) for i in range(len(a_train))]
test_data = [compute_statistics(a_test[i]) for i in range(len(a_test))]

print(train_data[0])
print(test_data[0])

features = ["Min", "Max", "Mean", "Median", "Standard Deviation", "Skewness", "Kurtosis"]
train_data = pd.DataFrame(train_data, columns=features)
test_data = pd.DataFrame(test_data, columns=features)

# %% [markdown]
# # Classification

# %% [markdown]
# ##### KNN

# %%
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_data, train_labels)
train_prediction = knn.predict(train_data)
train_accuracy = accuracy_score(train_labels, train_prediction)

# %%
test_prediction = knn.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_prediction)
# print(test_labels)
# print("\n\n\n")
# print(test_prediction)

cm = confusion_matrix(test_labels, test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels))
disp.plot(cmap="Purples")

print("KNN Results")
print(f"train accuracy = {train_accuracy}")
print(f"test accuracy = {test_accuracy}")

# %% [markdown]
# ##### SVM

# %%
svm_model = SVC(kernel='rbf', C=0.5, gamma='scale')
svm_model.fit(train_data, train_labels)

# %%
test_pred = svm_model.predict(test_data)
train_pred = svm_model.predict(train_data)

test_accuracy = accuracy_score(test_labels, test_pred)
train_accuracy = accuracy_score(train_labels, train_pred)


print("SVM Results")
print(f"train accuracy = {train_accuracy}")
print(f"test accuracy = {test_accuracy}")

cm = confusion_matrix(test_labels, test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels))
disp.plot(cmap="Purples")


# %% [markdown]
# ##### Random Forest Classifier 

# %%

rf = RandomForestClassifier(n_estimators=1, random_state=42,max_depth=1)
rf.fit(train_data, train_labels)

# Evaluate on the training set for Random Forest
train_prediction_rf = rf.predict(train_data)
train_accuracy_rf = accuracy_score(train_labels, train_prediction_rf)

# Evaluate on the test set for Random Forest
test_prediction_rf = rf.predict(test_data)
test_accuracy_rf = accuracy_score(test_labels, test_prediction_rf)

# Display results for Random Forest
print("Random Forest Results")
print(f"Train Accuracy: {train_accuracy_rf}")
print(f"Test Accuracy: {test_accuracy_rf}")

cm = confusion_matrix(test_labels, test_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels))
disp.plot(cmap="Purples")


# %% [markdown]
#  ##### Multilayer Perceptron Classifier

# %%
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=30, max_iter=300)
mlp.fit(train_data, train_labels)

train_prediction_mlp = mlp.predict(train_data)
train_accuracy_mlp = accuracy_score(train_labels, train_prediction_mlp)

test_prediction_mlp = mlp.predict(test_data)
test_accuracy_mlp = accuracy_score(test_labels, test_prediction_mlp)

print("Multilayer Perceptron (MLP) Classifier Results")
print(f"Train Accuracy: {train_accuracy_mlp}")
print(f"Test Accuracy: {test_accuracy_mlp}")


# %% [markdown]
#  ##### Linear Discriminant Analysis

# %%
lda = LinearDiscriminantAnalysis()
lda.fit(train_data, train_labels)

train_prediction_lda = lda.predict(train_data)
train_accuracy_lda = accuracy_score(train_labels, train_prediction_lda)

test_prediction_lda = lda.predict(test_data)
test_accuracy_lda = accuracy_score(test_labels, test_prediction_lda)

print("Linear Discriminant Analysis (LDA) Results")
print(f"Train Accuracy: {train_accuracy_lda}")
print(f"Test Accuracy: {test_accuracy_lda}")

# %% [markdown]
#  ##### Logistic Regression

# %%
log_reg = LogisticRegression(random_state=30, max_iter=200)
log_reg.fit(train_data, train_labels)

train_prediction_log_reg = log_reg.predict(train_data)
train_accuracy_log_reg = accuracy_score(train_labels, train_prediction_log_reg)

test_prediction_log_reg = log_reg.predict(test_data)
test_accuracy_log_reg = accuracy_score(test_labels, test_prediction_log_reg)

print("Logistic Regression Results")
print(f"Train Accuracy: {train_accuracy_log_reg}")
print(f"Test Accuracy: {test_accuracy_log_reg}")

# %%
import pickle

with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf, file)



