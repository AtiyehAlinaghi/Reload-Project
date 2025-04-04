############################################################################################
#
# It takes the extracted features and labels to perform SVM classifcation
# Then reports the performance and plots the ROC
#
############################################################################################
#
# Last modified by Atiyeh Alinaghi 04/04/2025 10:28 a.m.
#
# University of Southampton, ISVR
#
############################################################################################


import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Change the path to your features file.
features = pd.read_csv('same_demogs_cough_mix_features.csv')

# Change the path to your labels file.
labels = pd.read_csv('same_demogs_cough_labels.csv')


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# svm_model = SVC(kernel= 'linear')
# Create an SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
loo = LeaveOneOut()

# To store predictions and actuals
y_true, y_pred, y_probs = [], [], []

for train_ind, test_ind in loo.split(features_scaled):
    X_train, X_test = features_scaled[train_ind], features_scaled[test_ind]
    y_train, y_test = labels.iloc[train_ind], labels.iloc[test_ind]

    svm_model.fit(X_train,y_train.values.ravel())
    prediction = svm_model.predict(X_test)
    # Get probability for the positive class (class 1)
    prob = svm_model.predict_proba(X_test)  # Extract single probability

    y_pred.append(prediction[0])
    y_probs.append(prob[:,1])
    y_true.append(y_test.values[0][0])

# Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_probs)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)


print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
print("AUC:", auc_score)

# Plotting
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()