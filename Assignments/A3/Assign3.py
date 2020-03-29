import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
fig, (ax2,ax1) = plt.subplots(nrows=1,ncols=2,figsize=(8, 4))

#creating a dataset of 160 coin tosses with 60 heads(1) and 100 tails(0)
a=np.ones(60)
print(a)
b=np.zeros(100)
print(b)
data=np.concatenate((a,b))
np.random.shuffle(data)  #random mixing of data set
print(data)
print(np.average(data))

#declaring parameters of beta distribution to ensure prior has mean = .4
a=A=2
b=B=3

interval = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
y =  beta.pdf(interval, a, b)  

#Sequential Learning Approach
for i in range(len(data)):
    if(data[i] == 1):
        a+=1
        y =  beta.pdf(interval, a, b)
        ax1.plot(interval, y)

        
    else:
        b+=1
        y =  beta.pdf(interval, a, b)
        ax1.plot(interval, y)



A+=60
B+=100
Y= beta.pdf(interval,A,B)
ax2.plot(interval,Y)      
ax1.set_title('Seq\n$\\regular_{All\ 160\ iterations\ shown}$', fontsize=20)
ax2.set_title('Batch\n$\\regular_{Final\ plot\ shown}$', fontsize=20)

plt.show()
