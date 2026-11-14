import pandas as pd


def main():

	data = {"Name":["Sagar","Amit","Pooja"],
		"Age":[23,26,25],
		"City":["Pune","Mumbai","Satara"]
    }
	dobj = pd.DataFrame(data)
	
 
	
 #fetching first row data using loc
	print(dobj.loc[:,"Name"])
	print(dobj.loc[dobj['Age'] > 25, 'Name'])
 
	print("---------------------------------	")
	highest_value = dobj['Age'].max()
	print("Highest value : ",highest_value)
 
 #.iloc[-1]-iloc accesses values using their integer positions.
#-1 means the last element.The last element of the top two values 
	print("---------------------------------	")
	second_highest_value = dobj['Age'].nlargest(2).iloc[-1]
	print("Second highest value : ",second_highest_value)
 
	print("---------------------------------	")
	second_lowest_value = dobj['Age'].nsmallest(2).iloc[-1]
	print("Second lowest value : ",second_lowest_value)
 
 
 
if __name__ == "__main__":
    main()

	