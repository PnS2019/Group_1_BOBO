import os
#Rename.py
#1. Place this file into the project folder
#2. Create another folder called "data" in the same directory
#3. In the data folder, create some more folders called by the objects of the traindata
#4. Fill the folders with pictures of the corresponding objects in .jpg format
#5. Run this script ONCE =>$ python renamy.py
#6. every picture in the object folders will be renamed like the parent folder

#Select every folder ind data/...
for folder in os.listdir("data/"):
    if(folder != ".DS_Store"): #ignore this one

        #each folder contains pictures of the object written in the folder's name
        objectname = folder
        i = 0
        #rename each file and name it after the containing folder
        for filename in os.listdir("data/" + objectname + "/"):
            dst = objectname + str(i) + ".jpg"
            src = "data/" + objectname + "/" + filename
            dst ="data/" + objectname + "/" + dst
            os.rename(src, dst)
            i += 1
