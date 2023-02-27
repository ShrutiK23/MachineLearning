import sys
import pandas as pd
import numpy as np
import csv
import math


def sigmoid_o(w_h1,w_h2,w_h3,w_o,training_data,h1_net,h2_net,h3_net):
    #print(training_data)
    for i in range(len(w_h1)):
        h1_net += w_h1[i]*float(training_data[i])
    # print("h1_net = ",h1_net)
    for i in range(len(w_h2)):
        h2_net += w_h2[i]*float(training_data[i])
    #print("h2_net = ",h2_net)
    for i in range(len(w_h3)):
        h3_net += w_h3[i]*float(training_data[i])
    #print("h3_net = ",h3_net)
    

    sig_h1 = sigmoid_function_h(h1_net)
    sig_h2 = sigmoid_function_h(h2_net)
    sig_h3 = sigmoid_function_h(h3_net)
    sig_array = np.array([1,sig_h1,sig_h2,sig_h3])
    net_o = np.sum(w_o*sig_array)

    #print(net_o)
    sigmoid_output = sigmoid_function_h(net_o)
    
    return sigmoid_output,sig_h1,sig_h2,sig_h3,sig_array


def sigmoid_function_h(output):
    return (1/(1+np.exp(-output)))


if __name__ == "__main__":
    file_name = str(sys.argv[2])
    learning_rate = float(sys.argv[4]) 
    iterations = int(sys.argv[6]) 

    df_object = pd.read_csv(file_name,header=None)
    #df_object.set_axis(["a","b","t"],axis="columns",inplace = True)
    df_object.columns = ["a","b","t"]
    
    df_object.insert(loc=0, column='1', value=['1' for i in range(df_object.shape[0])])

    #print(df_object)
    #weights initialization
    w_h1 = [0.2,-0.3,0.4]   #w_bias_h1 = 0.2, w_a_h1 = -0.3, w_b_h1 = 0.4
    w_h2 = [-0.5,-0.1,-0.4]   #w_bias_h2 = -0.5, w_a_h2 = -0.1, w_b_h2 = -0.4
    w_h3 = [0.3,0.2,0.1]   #w_bias_h3 = 0.3, w_a_h3 = 0.2, w_b_h3 = 0.1
    w_o = [-0.1,0.1,0.3,-0.4]   #w_bias_o = -0.1, w_h1_o = 0.1, w_h2_o = 0.3, w_h3_o = -0.4
    #h_o = [0.0,0.0,0.0] #hidden layer output
    h1_net,h2_net,h3_net = 0.0,0.0,0.0
    w_o_net = np.empty(0)
    w_h1_net= np.empty(0)
    w_h2_net= np.empty(0)
    w_h3_net= np.empty(0)
    delta_h1 = 0.0

    print("-,-,-,-,-,-,-,-,-,-,-",*w_h1,*w_h2,*w_h3,*w_o,sep=',')

    outputFile = 'solution_' + file_name.split('.')[0] + '_learningRate_' + str(learning_rate) + '.csv'
    
    with open(outputFile, 'w', newline= '') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=' ')
        writer.writerow(['a','b','h1','h2','h3','o','t','delta_h1','delta_h2','delta_h3','delta_o','w_bias_h1','w_a_h1','w_b_h1',
        	'w_bias_h2','w_a_h2','w_b_h2','w_bias_h3','w_a_h3','w_b_h3','w_bias_o','w_h1_o','w_h2_o','w_h3_o'])
        writer.writerow(['-','-','-','-','-','-','-','-','-','-','-',*[val for val in w_h1],*[val for val in w_h2],*[val for val in w_h3],*[val for val in w_o]])
        for iteration in range(iterations):
            for j in range(len(df_object)):
                h1_net,h2_net,h3_net = 0.0,0.0,0.0
                w_h1_net = np.empty(0)
                w_h2_net= np.empty(0)
                w_h3_net= np.empty(0)
                w_o_net= np.empty(0)

                training_data = df_object.iloc[j, [df_object.columns.get_loc(c) for c in ['1','a', 'b']]]
                # if(j==1 and iteration == 0):
                #     print('training data before cal ',training_data[0],training_data[1],training_data[2])
                    
                sigmoid_output,sig_h1,sig_h2,sig_h3,sig_array = sigmoid_o(w_h1,w_h2,w_h3,w_o,training_data,h1_net,h2_net,h3_net)
                # print("sigmoid_output",sigmoid_output)
                #print(df_object.iloc[2])
                error_t_o = df_object.iloc[j][3]-sigmoid_output
                #print(str(df_object.iloc[j][3]),end='')
                if error_t_o:
                        
                    delta_o = sigmoid_output*(1-sigmoid_output)*(df_object.iloc[j][3]-sigmoid_output)
                    delta_h1 = sig_h1*(1-sig_h1)*(w_o[1]*delta_o)
                    delta_h2 = sig_h2*(1-sig_h2)*(w_o[2]*delta_o)
                    delta_h3 = sig_h3*(1-sig_h3)*(w_o[3]*delta_o)
                    
                    
                    
                    # if(j==1 and iteration == 0):
                    #     print("w_h1 = ",w_h1)
                    #     print("w_h2 = ",w_h2)
                    #     print("w_h3 = ",w_h3)
                    #     print("w_o = ",w_o)
                    #     print("a = ",df_object.iloc[j][1])
                    #     print("b = ",df_object.iloc[j][2])
                    #     print("sig =",sig_array)
                    #     print("sig_out =",sigmoid_output)
                    #     print("t = ",df_object.iloc[j][3])
                        
                    #     print("del_h1 =",delta_h1)
                    #     print("del_h2 =",delta_h2)
                    #     print("del_h3 =",delta_h3)
                    #     print("del_Out =",delta_o)
                    

                    for i in range(len(w_o)):
                        w_o[i] = w_o [i] + (learning_rate*sig_array[i]*delta_o)
                
                    for i in range(len(w_h1)):
                        w_h1[i] = w_h1[i] + (learning_rate*float(delta_h1)*float(df_object.iloc[j][i]))

                    for i in range(len(w_h2)):
                        w_h2[i] = w_h2[i] + (learning_rate*float(delta_h2)*float(df_object.iloc[j][i]))

                    for i in range(len(w_h3)):
                        w_h3[i] = w_h3[i] + (learning_rate*float(delta_h3)*float(df_object.iloc[j][i]))
                 
                    for no in range(len(w_h1)):
                        w_h1_net= np.append(w_h1_net,round(w_h1[no],5))
                    for no in range(len(w_h2)):
                        w_h2_net = np.append(w_h2_net,round(w_h2[no],5))
                    for no in range(len(w_h3)):
                        w_h3_net = np.append(w_h3_net,round(w_h3[no],5))
                    for no in range(len(w_o)):
                        w_o_net = np.append(w_o_net,round(w_o[no],5))


                    a1 = str(round(df_object.iloc[j][1],5))
                    a2 = str(round(df_object.iloc[j][2],5))

                    writer.writerow([a1,a2,*[str(round(sig_h1,5))],*[str(round(sig_h2,5))],*[str(round(sig_h3,5))],*[str(round(sigmoid_output,5))],*[str(round(df_object.iloc[j][3],5))],*[str(round(delta_h1,5))],*[str(round(delta_h2,5))],*[str(round(delta_h3,5))],*[str(round(delta_o,5))],*[val for val in w_h1_net],*[val for val in w_h2_net],*[val for val in w_h3_net],*[val for val in w_o_net]])
                    
                    print(a1,a2,*[str(round(sig_h1,5))],*[str(round(sig_h2,5))],*[str(round(sig_h3,5))],*[str(round(sigmoid_output,5))],*[str(round(df_object.iloc[j][3],5))],*[str(round(delta_h1,5))],*[str(round(delta_h2,5))],*[str(round(delta_h3,5))],*[str(round(delta_o,5))],*[val for val in w_h1_net],*[val for val in w_h2_net],*[val for val in w_h3_net],*[val for val in w_o_net])
                    
