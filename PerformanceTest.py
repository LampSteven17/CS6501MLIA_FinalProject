# Code for Calculating Accuracy, recall, precision and F1 score

from sklearn import metrics

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Predict and calculate 
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[1, 0]
FN = cm[0, 1]

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
auc = metrics.auc(fpr, tpr)

