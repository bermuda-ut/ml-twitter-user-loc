def mlTypeToName(mltype, knnMod=5):
    if(mltype == 0):
        name = "DTree"

    elif(mltype == 1):
        name = "NB"

    elif(mltype == 2):
        name = "MLP"

    elif(mltype == 3):
        name = "SVM"

    elif(mltype == 4):
        name = "KNN"+str(knnMod)

    #-- baseline --
    elif(mltype == 5):
        name = "Zero-R"

    elif(mltype == 6):
        name = "One-R"

    elif(mltype == 7):
        name = "Random"

    elif(mltype == 8):
        name = "Forest"

    else:
        return "ERROR"

    return name
