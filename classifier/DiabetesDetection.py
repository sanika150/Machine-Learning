import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from BaggingRegressionCalifornia import Y_pred, Y_test

def Classifier(DataPath):
    Border = "-"*40
    #------------------------------------------------------------
    #Step 1:Load the dataset from CSV file
    #------------------------------------------------------------
    print(Border)
    print("Step 1:Load the dataset from CSV file")
    print(Border)

    df=pd.read_csv(DataPath)
    print(Border)
    print("Some entries from dataset")
    print(df.head())
    print(Border)
    #------------------------------------------------------------
    #step 2:clean the dataset by removing empty rows
    #------------------------------------------------------------

    print(Border)
    print("step 2:clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace=True) #remove empty rows
    print("Total records : ",df.shape[0])
    print("TOtal columns : ",df.shape[1])
    print(Border)

    #------------------------------------------------------------
    #step 3:Separate independent and dependent variables
    #------------------------------------------------------------
    print(Border)
    print("step 3:Separate independent and dependent variables")
    print(Border)

    X = df.drop(columns = ['Outcome'])
    Y = df['Outcome']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(Border)
    print("Input columns : ",X.columns.to_list())
    print("Output column : Outcome")


    #------------------------------------------------------------
    #step 4:Split the data set for training and testing
    #------------------------------------------------------------
    print(Border)
    print("step 4:Split the data set for training and testing")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
    #stratify used for multile class related to random state
    #when we run class again data will not vary next time(no of data from label is fix)

    print(Border)
    print("Information of training and testing data")
    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)
    print(Border)

    #------------------------------------------------------------
    #step 5:Feature scaling
    #------------------------------------------------------------
    print(Border)
    print("step 5:Feature scaling")
    print(Border)

    #independent variable scaling
    scalar = StandardScaler()
    X_train_scaled= scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done")

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

    print("Classification report ")
    print(classification_report(Y_test,Y_pred))


def main():
    Border = "-"*40
    print(Border)
    print("Diabetes Classifier using KNN")
    print(Border)

    Classifier("diabetes.csv")




if __name__ == "__main__":
    main()
