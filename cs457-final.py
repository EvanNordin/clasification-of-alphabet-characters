# %%
#import libraries for working with files and directories
import random
import numpy as np
import pandas as pd


#we will be using scikit learn for machine learning

#our function that will add noise to the data
def createNoise(population):
    for i in range(1, len(population)):
        for j in range(population[i]):
            rand = random.randint(1,100)
            rand = rand / 100
            if(rand < 0.1):
                if(population[i] == 0):
                    population[i] = 1
                elif(population[i] == 1):
                    population[i] = 0

# def populateTargets(y):

#     n = len(y)/10

#     for i in range(len(y)):



#Defining the prototypes of our characters we want to classify. Prototybe A is at position 0, B is at position 1, and C is at position 2, and so on.
A = [0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,1,1]
B = [1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,1,1,1,1,1,0,0]
C = [0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0]
D = [1,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,0,0,0]
E = [1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,0]
F = [1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]
G = [0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0]
H = [1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1]
I = [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1]
J = [0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0]

#create a dataframe to store the data
df = pd.DataFrame(columns=['A','B','C','D','E','F','G','H','I','J'])
#add the prototypes to the dataframe
df['A'] = A
df['B'] = B
df['C'] = C
df['D'] = D
df['E'] = E
df['F'] = F
df['G'] = G
df['H'] = H
df['I'] = I
df['J'] = J

#Store the number of columns in the dataframe for later use
num_cols = len(df.columns)

#Ask the user for the nuumber of noisy samples they want to generate
n = int(input("How many samples? (integers > 1 only) "))

#Create n - 1 copies of each column in the dataframe and store it as a new dataframe
df_noisy = df.copy()
for col in df.columns:
    for j in range(n-1):
        df_noisy[col+str(j+1)] = df[col]

#store the number of columns in the noisy dataframe for later use
num_cols_noisy = len(df_noisy.columns)

#drop the first 10 columns from the dataframe
df_noisy = df_noisy.drop(df_noisy.columns[0:10], axis=1)


#iterate over each column in the dataframe and add noise to each column
for col in df_noisy.columns:
    createNoise(df_noisy[col])

# for i in range(len(df_noisy_list)):
#     #add noise to the dataframe
#     createNoise(df_noisy_list)

    
#create a new dataframe that combines the noisy dataframe and the original dataframe
df_combined = pd.concat([df, df_noisy], axis=1)

#write the dataframe to a csv file
df_combined.to_csv('cs457-final.csv', index=False)

#create a new dataframe the csv file. This will give us a dataframe that isnt fragmented
dataset = pd.read_csv('cs457-final.csv')


samples = []
for col in dataset.columns:
    samples.append(dataset[col])

#turn X into a numpy array
samples = np.array(samples).T

#we will now build the array of targets
targets = df.copy()
for i in range(n-1):
    temp = df.copy()
    #append temp to targets
    targets = pd.concat([targets, temp], axis=1)

#convert targets to a numpy array
targets = np.array(targets)

from sklearn.model_selection import train_test_split
samples_train, samples_test, targets_train, targets_test = train_test_split(samples, targets, test_size=0.40, random_state=1)

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, learning_rate='adaptive', learning_rate_init=0.01)

mlp_model.fit(samples_train, targets_train)

print("MLP Model Results\nThe average accuracy of MLP classification with N_SAMPLES=" + str(n) + " is: " + str(mlp_model.score(samples_test, targets_test)) + "\n\n")

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(samples_train, targets_train)

print("Random Forest Model Results\nThe average accuracy of Random Forest classification with N_SAMPLES=" + str(n) + " is: " + str(rf_model.score(samples_test, targets_test)) + "\n""\n")

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(samples_train, targets_train)

print("K-nearest neighbors model results\nThe average accuracy of KNN classification with N_SAMPLES=" + str(n) + " is: " + str(knn_model.score(samples_test, targets_test)) + "\n\n")


# from sklearn import svm
# svm_model = svm.SVC(kernel='linear', C=.1, gamma=1)
# svm_model.fit(samples_train, targets_train)


# print("SVM model results\nThe average accuracy of SVM classification with N_SAMPLES=" + str(n) + " is: " + str(svm_model.score(samples_test, targets_test)) + "\n\n")

test1 = np.array(A)
test2 = np.array(B)
test3 = np.array(C)
test4 = np.array(D)
test5 = np.array(E)
test6 = np.array(F)
test7 = np.array(G)
test8 = np.array(H)
test9 = np.array(I)
test10 = np.array(J)

#get the eigenvectors and eigenvalues from the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(np.cov(samples_train))

print(eigenvalues)
print(eigenvectors)



print("The dot product of A and A transposed is: " + str(np.dot(test1, test1.T)))
print("The dot product of B and B transposed is: " + str(np.dot(test2, test2.T)))
print("The dot product of C and C transposed is: " + str(np.dot(test3, test3.T)))
print("The dot product of D and D transposed is: " + str(np.dot(test4, test4.T)))
print("The dot product of E and E transposed is: " + str(np.dot(test5, test5.T)))
print("The dot product of F and F transposed is: " + str(np.dot(test6, test6.T)))
print("The dot product of G and G transposed is: " + str(np.dot(test7, test7.T)))
print("The dot product of H and H transposed is: " + str(np.dot(test8, test8.T)))
print("The dot product of I and I transposed is: " + str(np.dot(test9, test9.T)))
print("The dot product of J and J transposed is: " + str(np.dot(test10, test10.T)))


# %%

# %%
