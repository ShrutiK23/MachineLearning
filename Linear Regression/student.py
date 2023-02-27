import sys
import csv
import numpy

if __name__ == "__main__":
    file_name = str(sys.argv[2])
    eta = float(sys.argv[4]) #learningRate
    threshold = float(sys.argv[6])

    with open(file_name) as csvFile:
            reader = csv.reader(csvFile, delimiter=',')
            X = []
            Y_Actual = []
            for row in reader:
                X.append(row[:-1])
                Y_Actual.append([row[-1]])

    n = len(X)
    X = numpy.array(X).astype(float)
    
    Y_Actual = numpy.array(Y_Actual).astype(float)
    W = numpy.zeros((X.shape[1], n), dtype=float)
    w0 = numpy.zeros((1, n), dtype=float)
    Y_Pred = numpy.zeros((n, X.shape[1]), dtype=float)

    for i in range(X.shape[1]):
        for j in range(n):
            Y_Pred[j][i] = Y_Pred[j][i] + W[i][j]*X[j][i]

    Y_Pred = w0 + Y_Pred.sum(axis=1)
    Y_Pred = numpy.reshape(Y_Pred, (n,1))

    Z = Y_Actual - Y_Pred
    grad_temp = numpy.multiply(X, Z)
    grad = numpy.sum(grad_temp, axis=0)
    grad1 = numpy.sum(Z, axis=0)

    W_init = W
    W_init = numpy.append(w0, W, axis=0)

    for i in range(n):
        W[0][i] = W[0][i] + (eta*grad[0])
        W[1][i] = W[1][i] + (eta*grad[1])

        w0[0][i] = w0[0][i] + (eta*grad1)

    Weight = numpy.append(w0, W, axis=0)

    SSE_old = numpy.sum(numpy.square(Y_Actual - Y_Pred))

    outputFile = 'solution_' + file_name.split('.')[0] + '_learningRate_' + str(eta) + 'threshold' + str(threshold) + '.csv'

    with open(outputFile, 'w', newline= '') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='')
        writer.writerow([*[0], *[val for val in W_init.T[0]], *[SSE_old]])
        print(*[0], *[val for val in W_init.T[0]], *[SSE_old])
       
        iteration = 1
        while True:
            Y_Pred = numpy.zeros((n, X.shape[1]), dtype= float)
            for i in range(X.shape[1]):
                for j in range(n):
                    Y_Pred[j][i] = Y_Pred[j][i] + W[i][j]*X[j][i]

            Y_Pred = w0 + Y_Pred.sum(axis=1)
            Y_Pred = numpy.reshape(Y_Pred, (n,1))

            SSE_new = numpy.sum(numpy.square(Y_Actual - Y_Pred))

            if abs(SSE_old - SSE_new) > threshold:
                writer.writerow([*[iteration], *[val for val in Weight.T[0]], *[SSE_new]])
                print(*[iteration], *[val for val in Weight.T[0]], *[SSE_new])
                Z = Y_Actual - Y_Pred
                grad_temp = numpy.multiply(X, Z)
                grad = numpy.sum(grad_temp, axis=0)
                grad1 = numpy.sum(Z, axis=0)
                for i in range(n):
                    W[0][i] = W[0][i] + (eta*grad[0])
                    W[1][i] = W[1][i] + (eta*grad[1])

                    w0[0][i] = w0[0][i] + (eta*grad1)

                Weight = numpy.append(w0, W, axis=0)
                iteration += 1
                SSE_old = SSE_new
            else:
                break
        writer.writerow([*[iteration], *[val for val in Weight.T[0]], *[SSE_new]])
        print(*[iteration], *[val for val in Weight.T[0]], *[SSE_new])