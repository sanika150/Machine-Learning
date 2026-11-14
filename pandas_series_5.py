import pandas as pd


def main():
    

	sobj = pd.Series([25000,27000,29000,30000],index=['C','Java','Python','React'])
 


	print(sobj)
	print(sobj['Python'])

    
if __name__ == "__main__":
    main()

	