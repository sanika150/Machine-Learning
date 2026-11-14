from sklearn.datasets import load_iris


def main():
	print("Iris classification case study")

	Dataset = load_iris()

	print("Independent variables are:")

	print(Dataset.feature_names)
	print("Length of independent variables :",len(Dataset.feature_names))

	print("Dependent variables are:")
	print(Dataset.target_names)
	print("length of dependent variables:",len(Dataset.target_names))
if __name__ == "__main__":
    main()

	