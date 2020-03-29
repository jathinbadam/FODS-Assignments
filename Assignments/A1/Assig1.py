import numpy as np
import pandas as pd
from math import sqrt

class Setup:

    def __init__(self, degree):
        self.exp = []
        for i in range(degree+1):
            for j in range(degree+1):
                if i+j <= degree:
                    self.exp.append((i, j)) #Forming tuples of powers of features

    def data_split(self, dataframe):
        normalize = lambda x: ((x - x.min()) / (x.max() - x.min()))
        self.data = pd.DataFrame([])  #New DataFrame to add the newly generated colums of different powers of features
        self.count = -1
        for (a, b) in self.exp:
            self.count += 1
            res = ((dataframe["latitude"] ** a) * (dataframe["longitude"] ** b))
            self.data.insert(self.count, "col" + str(a) + str(b), res, True)

        self.count += 1 
        dataframe = normalize(dataframe) #Normalizing the data
        self.data = normalize(self.data) #Normalizing the data
        self.data["col00"] = [1.0]*len(self.data)
        
        # generate a 70-20-10 split on the data:  
        # Splitting the data into test, validation and train data.
        X = self.data[:304000]
        Y = dataframe["altitude"][:304000]
        xval = self.data[304000:391088]
        yval = dataframe["altitude"][304000:391000]
        x = self.data[391000:]
        y = dataframe["altitude"][391000:]   
        return (X, Y, xval, yval, x, y)

class Models:
    def __init__(self, N, X, Y, x, y, xval, yval)

        self.N = N
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.x = np.array(x)
        self.y = np.array(y)
        self.xval = np.array(xval)
        self.yval = np.array(yval)

    def errors(self, weights):  #Defination of R2 AND RMSE Errors.
        total = sum(np.square(np.mean(self.y) - self.y))
        result = sum(np.square((self.x @ weights) - self.y))
        rmse = sqrt(result/len(self.x))
        r2 = (1-(result / total))
        return [r2*100, rmse]

    def gradient_descent(self):
        learning_rate = 8e-7
        prev_error, count = 1e10, 0
        W = np.random.randn(self.N)
        while True:
            hypothesis = ((self.X @ W) - self.Y) #Main hypothesis function
            error = 0.5 * (hypothesis @ hypothesis) #cost function
            grad = (self.X.T @ hypothesis)
            if count % 2000 == 0:
                    print("epoch =", count, "| error_hypothesis =", prev_error-error)
                    print("error = ", error, "||", W)
                    print("errors =", self.errors(W), end="\n\n")
            W -= learning_rate * grad
            if abs(prev_error-error) <= 1e-5:  #Break condition
                break
            prev_error = error
            count += 1
        print(count, error)
        print(W, self.errors(W), end="\n\n")

    def sgd(self, epochs): #Method for Stochastic Gradient Descent
        learning_rate = 0.05
        W = np.random.randn(self.N)
        for count in range(epochs):
            hypothesis = ((self.X @ W) - self.Y)
            error = 0.5 * (hypothesis @ hypothesis)
            W -= learning_rate * (((self.X[count] @ W) - self.Y[count]) * self.X[count])
            if count % 500 == 0:
                print("epoch =", count)
                print("error =", error, "||", W)
                print("errors =", self.errors(W), end="\n\n")

    def gd_L1(self):     #Method for L1 normalization 
        W_fin = np.array([])
        learning_rate, l1_fin = 5e-7, 0
        min_val_loss = 1e10
        L1_coefficients = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45,0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
        sgn = lambda x: (x / abs(x)) #signum function
        for l1 in L1_coefficients:
            prev_error, count = 1e10, 0
            W = np.random.randn(self.N)
            while True:
                hypothesis = ((self.X @ W) - self.Y)
                error = 0.5 * ((hypothesis @ hypothesis) + l1*sum([abs(w) for w in W]))
                if count % 500 == 0:
                    print("L1 hyperparamter =", l1, end=", ")
                    print("epoch =", count, "| error_hypothesis =", prev_error-error)
                    print("error = ", error, "||", W)
                    print("errors =", self.errors(W), end="\n\n")
                sgn_w = np.array([sgn(w) for w in W])
                W -= learning_rate * ((self.X.T @ hypothesis) + 0.5*l1*sgn_w)
                if abs(prev_error-error) <= 0.005:
                    break
                prev_error = error
                count += 1
            val_D = ((self.xval @ W) - self.yval)
            val_loss = 0.5 * ((val_D.T @ val_D) + l1*sum([abs(w) for w in W]))
            if val_loss < min_val_loss:
                W_fin = W
                l1_fin = l1
                min_val_loss = val_loss
        print(min_val_loss, l1_fin, W_fin)

    def gd_L2(self):   #Method for L2 normalization 
        W_fin = np.array([])
        learning_rate, l2_fin = 5e-7, 0
        min_val_loss = 1e10
        L2_vals = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45,
                   0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
        for l2 in L2_vals:
            prev_error, count = 1e10, 0
            W = np.random.randn(self.N)
            while True:
                hypothesis = ((self.X @ W) - self.Y)
                error = 0.5 * ((hypothesis @ hypothesis) + l2*sum([w*w for w in W]))
                if count % 500 == 0:
                    print("L2 hyperparamter =", l2, end=", ")
                    print("epoch =", count, "| error_hypothesis =", prev_error-error)
                    print("error = ", error, "||", W)
                    print("errors =", self.errors(W), end="\n\n")
                W -= learning_rate * ((self.X.T @ hypothesis) + l2*W)
                if abs(prev_error-error) <= 0.005:
                    break
                prev_error = error
                count += 1
            val_D = ((self.xval @ W) - self.yval)
            val_loss = 0.5 * ((val_D.T @ val_D) + l2 * (W.T @ W))
            if val_loss < min_val_loss:
                W_fin = W
                l2_fin = l2
                min_val_loss = val_loss
        print(min_val_loss, l2_fin, W_fin)
    
    def normalEquations(self):
	y =  np.array(data.altitude)
	x1=np.array(data.latitude)
	x2=np.array(data.longitude)
	m=y.size
	x1=np.reshape(x1,(m,1))
	x2=np.reshape(x2,(m,1))
	x=np.append(x1,x2,axis=1)
	x_bias = np.ones((m,1))
	x = np.append(x_bias,x,axis=1)
	x_transpose = np.transpose(x)
	x_transpose_dot_x = x_transpose.dot(x)
	temp_1 = np.linalg.inv(x_transpose_dot_x)
	temp_2=x_transpose.dot(y)
	theta =temp_1.dot(temp_2)
	print(theta)

    def check(self):
        B = self.X.T @ self.Y
        A = self.X.T @ self.X
        W = (np.linalg.inv(A)) @ B
        print(W, self.errors(W))

columns = ["useless", "latitude", "longitude", "altitude"]
raw_df = pd.read_csv("3D_spatial_network.txt", sep=',', header=None,names=columns).drop("useless", 1).sample(frac=1)
preprocessor = Setup(degree=1)
X_train, Y_train, x_val, y_val, x_test, y_test = preprocessor.data_split(raw_df)

model = Models(N=preprocessor.count,
                        X=X_train,
                        Y=Y_train,
                        x=x_test,
                        y=y_test,
                        xval=x_val,
                        yval=y_val)

model.check()
model.gradient_descent()
model.gd_L1()
model.gd_L2()
model.sgd()
