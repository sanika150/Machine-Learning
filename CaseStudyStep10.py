import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40
###########################################################
#Step 1: Load the dataset
###########################################################
print(Border)
print("Step 1 : load the dataset")
print(Border)

DatasetPath = "diabetes.csv"
#store csv data in df
df = pd.read_csv(DatasetPath)

print("Dataset gets loaded successfully...")
print("Intials entries from dataset : ")
print(df.head())

###########################################################
#Step 2: Data Analysis(EDA)
###########################################################
print(Border)
print("Step 2 : Data Analysis")
print(Border)

print("Shape of dataset : ",df.shape)
#list of column
print("Column Names : ",list(df.columns))
print("Missing values (Per column)")
print(df.isnull().sum())

print("Class Distribution (Outcome count)")
print(df["Outcome"].value_counts())

print("Statistical report of dataset :")
print(df.describe())

###########################################################
#Step 3: Decide independent and dependent vatiables
###########################################################
print(Border)
print("Step 3 : Decide independent and dependent vatiables")
print(Border)

# X: Independetn variables or features
# Y: Dependent variables or Labels

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[feature_cols]
Y = df["Outcome"]

print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)

###########################################################
#Step 4: Visualization of the dataset
###########################################################
print(Border)
print("Step 4 : Visualization of the dataset")
print(Border)

#Scatter plot
plt.figure(figsize=(7,5))

for sp in df["Outcome"].unique():
    temp = df[df["Outcome"] == sp]
    plt.scatter(temp["pregnancies"], temp["glucose"], label =sp)

plt.title("Diabetes : Pregnancies vs Glucose")

plt.xlabel("Pregnancies")
plt.ylabel("Glucose")
plt.legend()
plt.grid(True)
plt.show()

###########################################################
#Step 5: Split the dataset for training and testing
###########################################################
print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

#Test size = 20%
#Train size = 80%

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.5,
    random_state=42
)

print("Data splitting activity done : ")

print("X - Independent : ",X.shape) #(150,4)
print("Y - Dependent : ",Y.shape) #(150,)

print("X_train : ",X_train.shape) #(120,4)
print("X_test : ",X_test.shape) #(30,4)

print("Y_train : ",Y_train.shape) #(120,)
print("Y_test : ",Y_test.shape) #(30,)

###########################################################
#Step 6: Build the model
###########################################################
print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are goinf to use DecisionTreeClassifier ")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth= 5, #tunning tunned value
    random_state= 42
)

print("Model successfully created : ",model)

###########################################################
#Step 7: Train the model
###########################################################
print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)

print("Model training completed")

###########################################################
#Step 8: Evaluate the model
###########################################################
print(Border)
print("Step 8 : Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model Evaluation (Testing) complete")

print(Y_pred.shape)

print("Expected answers : ")
print(list[Y_test])

print("Ptredicted answers : ")
print(Y_pred)

###########################################################
#Step 9: Evaluate the model performance
###########################################################
print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)

print("Accuracy of model is : ",accuracy * 100)

cm = confusion_matrix(Y_test,Y_pred)

print("Confusion matrix : ")
print(cm)

print("Classification report ")
print(classification_report(Y_test,Y_pred))

###########################################################
#Step 10: Plot confusion matrix
###########################################################
print(Border)
print("Step 10 : Plot confusion matrix")
print(Border)

data = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)

data.plot()
plt.title("Confusion matrix of Iris dataset")
plt.show()