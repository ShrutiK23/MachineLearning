import sys
import pandas as pd
import numpy as np
import csv
import math


def Cal_W_I(nearest_neighbor,k,i,case):
    dist_k = euclidean_distance(case, nearest_neighbor[k - 1])
    d1 = euclidean_distance(case, nearest_neighbor[0])
    if dist_k == d1:
        return 1
    else:
        di = euclidean_distance(case, nearest_neighbor[i])
        return (dist_k - di) / (dist_k - d1)


def Cal_Weights(nearest_neighbor,k,case):
    weight_array = np.array([])
    for i in range(k):
        weight_i = Cal_W_I(nearest_neighbor,k,i,case)
        weight_array = np.append(weight_array, [weight_i])
    return weight_array

def Cal_KNN(casebase_array,case,k):
    nearest_neighbor = np.empty([0, 3])
    for i in range(len(casebase_array)):
        if len(nearest_neighbor) < k:
            nearest_neighbor = np.vstack((nearest_neighbor, casebase_array[i]))
            nearest_neighbor = sorted(
                nearest_neighbor,
                key=lambda x: math.sqrt(math.pow((case[1] - x[1]), 2) + math.pow((case[2] - x[2]),2)),
            )
        else:
            if euclidean_distance(case, casebase_array[i]) < euclidean_distance(case, nearest_neighbor[k - 1]):
                nearest_neighbor = np.vstack((nearest_neighbor, casebase_array[i]))
                nearest_neighbor = sorted(
                    nearest_neighbor,
                    key=lambda x: math.sqrt(
                        math.pow((case[1] - x[1]),2) + math.pow((case[2] - x[2]),2)
                    ),
                )
                nearest_neighbor = nearest_neighbor[:k]
    return nearest_neighbor

def euclidean_distance(x,y):

    dist = math.sqrt((math.pow((x[1]-y[1]),2))+(math.pow((x[2] - y[2]),2)))
    return dist

if __name__ == "__main__":
    file_name = str(sys.argv[2])
    k = int(sys.argv[4]) 

    df_object = pd.read_csv(file_name,header=None)

    
    casebase_array = np.empty([0, 3], dtype=float)
    classify_cases = np.empty([0, 3], dtype=float)
    absolute_errors = 0

    with open(file_name) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if row[0] == "A":
                row[0] = int(1) 
            else:
                row[0] = int(2)
            row[1] = float(row[1])
            row[2] = float(row[2])
        
        
            if casebase_array.size == 0:
                casebase_array = np.vstack([casebase_array, row])

            nearest_neighbor = Cal_KNN(casebase_array,row,1)

            if nearest_neighbor[0][0] != row[0]:
                casebase_array = np.vstack([casebase_array, row])
            else:
                classify_cases = np.vstack([classify_cases, row])

  
    for case in classify_cases:
        case_class=0
        case_weights = np.zeros(2)
        nearest_neighbor = Cal_KNN(casebase_array,case,k)
        weights = Cal_Weights(nearest_neighbor,k,case)
        for i in range(k):
            if nearest_neighbor[i][0] == 1:
                case_weights[0] += weights[i]
            else:
                case_weights[1] += weights[i]

        if (case_weights[0]>case_weights[1]):
            case_class = 1
        else:
            case_class = 2

        if case_class != case[0]:
            absolute_errors += 1

    print(absolute_errors)

    outputFile = 'solution_' + file_name.split('.')[0] +'.csv'

    with open(outputFile, 'w', newline= '') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=',')
        writer.writerow([absolute_errors])
        for case in casebase_array:
            result = "A" if case[0] == 1 else "B"
            writer.writerow([result,str(case[1]),str(case[2])])
            print(result,str(case[1]),str(case[2]))
            