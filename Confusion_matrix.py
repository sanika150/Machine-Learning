from sklearn.metrics import confusion_matrix

#Scikit-learn (sklearn) is a free, open-source Python machine learning library built on top of NumPy, SciPy, and Matplotlib. 
def main():
	Actual = [1,0,1,1,1,0,1,0,0,1]
	Predicted = [1,0,0,1,1,1,1,1,0,1]

	print("Actual data: ",Actual)
	print("Predicted data :",Predicted)

	con_mat = confusion_matrix(Actual,Predicted)

	print(con_mat)
 
if __name__ == "__main__":
    main()

	