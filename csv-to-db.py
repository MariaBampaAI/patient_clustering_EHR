import os
import sys
from glob import glob
import pandas as pd
from sqlalchemy import create_engine
import pathlib


#paste in input the directory of MIMIC
#input_directory = pathlib.Path(input('Please specify input folder of data: '))
#C:\\Users\\Maria\\Desktop\\Work\\data\\MIMICIII\\data\\mimic-iii-clinical-database-1.4
#if not input_directory.is_dir():
    #print("Input is invalid.  Bail or ask for a new input.")

#establish connection
DATABASE_NAME = "mimic3.db"
THRESHOLD_SIZE = 5 * 10 ** 7
CHUNKSIZE = 10 ** 6
CONNECTION_STRING = "sqlite:///{}".format(DATABASE_NAME)


engine = create_engine(CONNECTION_STRING, echo=False)
# place the csv files in the same working directory, the db will be saved there
print("Current working directory: {0}".format(os.getcwd()))


if os.path.exists(DATABASE_NAME):
    msg = "File {} already exists.".format(DATABASE_NAME)
    print(msg)
    sys.exit()
    

for f in glob("*.csv.gz"):
    #input_directory.
    print("Starting processing {}".format(f))
    f = str(f)

    # Change the current working directory to save the files in your data folder
    print("Current working directory: {0}".format(os.getcwd()))
    # C:\\Users\\Maria\\Desktop\\Work\\data\\MIMICIII\\data\\mimic-iii-clinical-database-1.4
    
    if os.path.getsize(f) < THRESHOLD_SIZE:
        df = pd.read_csv(f, index_col="ROW_ID")
        print(df)
        df.to_sql(f.strip(".csv.gz").lower(), CONNECTION_STRING)
        print(df)
    else:
        # If the file is too large, let's do the work in chunks
        for chunk in pd.read_csv(f, index_col="ROW_ID", chunksize=CHUNKSIZE):
            chunk.to_sql(
                f.strip(".csv.gz").lower(), CONNECTION_STRING, if_exists="append"
                )
    print("Finished processing {}".format(f))

print("Should be all done!")