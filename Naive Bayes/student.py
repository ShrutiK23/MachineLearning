import sys
import pandas as pd
import math

def Cal_PriorProbabilities(df_object):
    total_instances = len(df_object.index)
    
    Aclass = df_object.apply(lambda x : True
            if x['class'] == "A" else False, axis = 1)
  
    AclassCount = len(Aclass[Aclass == True].index)

    Bclass = df_object.apply(lambda x : True
            if x['class'] == "B" else False, axis = 1)
  
    BclassCount = len(Bclass[Bclass == True].index)

    priorProbA = AclassCount/total_instances
    priorProbB = BclassCount/total_instances
    
    return total_instances,AclassCount,priorProbA,BclassCount,priorProbB

def Calculate_Mean(df_object):
    meanAatt1 = df_object.loc[df_object['class'] == 'A', 'att1'].mean()
    meanAatt2 = df_object.loc[df_object['class'] == 'A', 'att2'].mean()
    
    meanBatt1 = df_object.loc[df_object['class'] == 'B', 'att1'].mean()
    meanBatt2 = df_object.loc[df_object['class'] == 'B', 'att2'].mean()
    
    return meanAatt1,meanAatt2,meanBatt1,meanBatt2

def Calculate_Variance(df_object,meanAatt1,meanAatt2,AclassCount,meanBatt1,meanBatt2,BclassCount):
    sumAatt1=0.0
    varianceAatt1 = 0.0
    sumAatt2=0.0
    varianceAatt2 = 0.0
    sumBatt1=0.0
    varianceBatt1 = 0.0
    sumBatt2=0.0
    varianceBatt2 = 0.0

    classArows = df_object[df_object.values  == "A"]
   
    for rows in classArows.index:
        sumAatt1 = sumAatt1 + (pow((classArows['att1'][rows] - meanAatt1),2))
        varianceAatt1 = sumAatt1 / (AclassCount-1)
        sumAatt2 = sumAatt2 + (pow((classArows['att2'][rows] - meanAatt2),2))
        varianceAatt2 = sumAatt2 / (AclassCount-1)
   

    classBrows = df_object[df_object.values  == "B"]
    for rows in classBrows.index:
        sumBatt1 = sumBatt1 + (pow((classBrows['att1'][rows] - meanBatt1),2))
        varianceBatt1 = sumBatt1 / (BclassCount-1)
        sumBatt2 = sumBatt2 + (pow((classBrows['att2'][rows] - meanBatt2),2))
        varianceBatt2 = sumBatt2 / (BclassCount-1)

    return varianceAatt1,varianceAatt2,varianceBatt1,varianceBatt2

def Calculate_Misclassfied(df_object,meanAatt1,varianceAatt1,meanAatt2,varianceAatt2,priorProbA,meanBatt1,varianceBatt1,meanBatt2,varianceBatt2,priorProbB):
    misclassifiedCount = 0
   
    for rows in df_object.index:
        prob_att1_A = (1/(math.sqrt(2*math.pi*varianceAatt1)))*(math.exp(-( (pow((df_object['att1'][rows] - meanAatt1),2))/(2*varianceAatt1) )))
        prob_att2_A = 1/(math.sqrt(2*math.pi*varianceAatt2))*math.exp(-( (pow((df_object['att2'][rows] - meanAatt2),2))/(2*varianceAatt2) ))

        prob_att1_B = 1/(math.sqrt(2*math.pi*varianceBatt1))*math.exp(-( (pow((df_object['att1'][rows] - meanBatt1),2))/(2*varianceBatt1) ))
        prob_att2_B = 1/(math.sqrt(2*math.pi*varianceBatt2))*math.exp(-( (pow((df_object['att2'][rows] - meanBatt2),2))/(2*varianceBatt2) ))

        prob_A = (priorProbA*prob_att1_A*prob_att2_A)/((priorProbA*prob_att1_A*prob_att2_A)+(priorProbB*prob_att1_B*prob_att2_B))
        prob_B = (priorProbB*prob_att1_B*prob_att2_B)/((priorProbA*prob_att1_A*prob_att2_A)+(priorProbB*prob_att1_B*prob_att2_B))
      
        if(prob_A > prob_B):
            if(df_object['class'][rows] != "A"):
                misclassifiedCount+=1
        else:
            if(df_object['class'][rows] != "B"):
                misclassifiedCount+=1
    print(misclassifiedCount)

if __name__ == "__main__":
    file_name = str(sys.argv[2])

    df_object = pd.read_csv(file_name,header=None)
    df_object.columns = ['class','att1','att2']

    #Step 1: calculate prior probabilities of A and B class
    total_instances,AclassCount,priorProbA,BclassCount,priorProbB = Cal_PriorProbabilities(df_object)

    #Step 2: calculate mean and variance of att1 and att2
    meanAatt1,meanAatt2,meanBatt1,meanBatt2 = Calculate_Mean(df_object)
    varianceAatt1,varianceAatt2,varianceBatt1,varianceBatt2 = Calculate_Variance(df_object,meanAatt1,meanAatt2,AclassCount,meanBatt1,meanBatt2,BclassCount)

    print(meanAatt1,varianceAatt1,meanAatt2,varianceAatt2,priorProbA)
    print(meanBatt1,varianceBatt1,meanBatt2,varianceBatt2,priorProbB)

    #Step 3: Calculate Misclassfied instances
    misclassifiedDataCount = Calculate_Misclassfied(df_object,meanAatt1,varianceAatt1,meanAatt2,varianceAatt2,priorProbA,meanBatt1,varianceBatt1,meanBatt2,varianceBatt2,priorProbB)
