from sklearn import tree

#rough = 1
#smooth = 0

#cricket = 2
#Tennis = 1

def main():
    print("Ball classification case study")

    X = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [96, 0], [43, 1], [110, 0], [35, 1], [95, 0]]

    # Dependent Variables
    Y = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    # independent variables for training
    Xtrain = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [96, 0], [43, 1], [110, 0]]

    # independent variables for testing
    Xtest = [[35, 1], [95, 0]]

    # Dependent variables for training
    Ytrain = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2]

    # Dependent variables for testing
    Ytest = [1, 2]

    modelobj = tree.DecisionTreeClassifier()
    trainmodel = modelobj.fit(Xtrain, Ytrain)
    result = trainmodel.predict([[35, 1]])
    print(type(result))

    if result == 1:
        print("Object looks like Tennis ball")
    elif result == 2:
        print("Object looks like Cricket ball")

if __name__ == "__main__":
    main()

	