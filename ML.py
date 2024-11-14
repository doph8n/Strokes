import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


data = pd.read_csv("processed.csv")

Y = data['stroke']
X = data.drop(['stroke'], axis = 1)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=99)


# Decision Tree with class weights
#classifier = DecisionTreeClassifier(class_weight="balanced")
#classifier.fit(X_train, Y_train)
#Y_pred_tree = classifier.predict(X_test)

# Random Forest Classifier without class weights
rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train,Y_train)
rf_smote = RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")
rf_smote.fit(X_resampled, Y_resampled)
Y_pred_rf_smote = rf_smote.predict(X_test)

# Metrics for Decision Tree
#accuracy_tree = accuracy_score(Y_test, Y_pred_tree)
#precision_tree = precision_score(Y_test, Y_pred_tree)
#recall_tree = recall_score(Y_test, Y_pred_tree)
#f1score_tree = f1_score(Y_test, Y_pred_tree)

#print("Decision Tree Classifier:")
#print(f"Precision = {precision_tree}")
#print(f"Recall = {recall_tree}")
#print(f"F1 Score = {f1score_tree}")
#print("Accuracy:", accuracy_tree)
#print("\n")

# Metrics for Balanced Random Forest
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
precision_rf = precision_score(Y_test, Y_pred_rf)
recall_rf = recall_score(Y_test, Y_pred_rf)
f1score_rf = f1_score(Y_test, Y_pred_rf)

conf_matrix_smote = confusion_matrix(Y_test, Y_pred_rf)
class_report_smote = classification_report(Y_test, Y_pred_rf)

print("Confusion Matrix:")
print(conf_matrix_smote)
print("\nClassification Report:")
print(class_report_smote)

print("Random Forest Classifier:")
print(f"Precision = {precision_rf}")
print(f"Recall = {recall_rf}")
print(f"F1 Score = {f1score_rf}")
print("Accuracy:", accuracy_rf)

print("\n")

# Metrics for Balanced Random Forest Smote
accuracy_rf_smote = accuracy_score(Y_test, Y_pred_rf_smote )
precision_rf_smote = precision_score(Y_test, Y_pred_rf_smote)
recall_rf_smote = recall_score(Y_test, Y_pred_rf_smote)
f1score_rf_smote = f1_score(Y_test, Y_pred_rf_smote)

print("Random Forest Classifier:")
print(f"Precision = {precision_rf_smote}")
print(f"Recall = {recall_rf_smote}")
print(f"F1 Score = {f1score_rf_smote}")
print("Accuracy:", accuracy_rf_smote)

conf_matrix_smote = confusion_matrix(Y_test, Y_pred_rf_smote)
class_report_smote = classification_report(Y_test, Y_pred_rf_smote)

print("Confusion Matrix:")
print(conf_matrix_smote)
print("\nClassification Report:")
print(class_report_smote)

# Plot Decision Tree (uncomment if you'd like to visualize)
#plt.figure(figsize=(20, 10))
#plot_tree(classifier, feature_names=data.columns, filled=True, class_names=['No Stroke', 'Stroke'])
#plt.show()

