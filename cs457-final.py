# %%
#import libraries for working with files and directories
import random
import numpy as np
import pandas as pd
import time


#we will be using scikit learn for machine learning

#our function that will add noise to the data
def createNoise(population, noise):
    noise = noise/100
    for i in range(1, len(population)):
        for j in range(population[i]):
            rand = random.randint(1,100)
            rand = rand / 100
            if(rand < noise):
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

#prompt the user for sample number and noise factor
n = int(input("How many samples do you want to generate? "))
noise = float(input("What is the noise factor (in percent)? "))

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
    createNoise(df_noisy[col], noise)

# for i in range(len(df_noisy_list)):
#     #add noise to the dataframe
#     createNoise(df_noisy_list)

    
#create a new dataframe that combines the noisy dataframe and the original dataframe
df_combined = pd.concat([df, df_noisy], axis=1)

#write the dataframe to a csv file
df_combined.to_csv('cs457-final.csv', index=False)

#create a new dataframe the csv file. This will give us a dataframe that isnt fragmented
dataset = pd.read_csv('cs457-final.csv')

# (dataset)

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

# print("Target\n\n" + str(targets))

#convert targets to a numpy array
targets = np.array(targets)

from sklearn.model_selection import train_test_split
samples_train, samples_test, targets_train, targets_test = train_test_split(samples, targets, test_size=0.40, random_state=1)



print("Beginning MLP Classification...\n")
from sklearn.neural_network import MLPClassifier
time_start = time.perf_counter()
mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, learning_rate_init=0.01)
mlp_model.fit(samples_train, targets_train)
time_end = time.perf_counter()

print("MLP Model Results\nThe average accuracy of MLP classification with N_SAMPLES=" + str(n) + " and noise = " + str(noise) +  "% is: " + str(100*mlp_model.score(samples_test, targets_test)) + "%.\nTraining completed in " + str(time_end - time_start) + " seconds.\n")


#optional part: optimizing the hyperparameters for random forest
# we used this to find the best parameters for the random forest model
# on line 143.
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingGridSearchCV
# import pandas as pd

# param_grid = {'max_depth': [3, 5, 10],
#               'min_samples_split': [2, 5, 10]}
# base_estimator = RandomForestClassifier(random_state=0)
# sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
#                          factor=2, resource='n_estimators',
#                          max_resources=30).fit(samples_train, targets_train)
# print(sh.best_estimator_)

print("Beginning Random Forest Classification...\n")
from sklearn.ensemble import RandomForestClassifier
time_start = time.perf_counter()
rf_model = RandomForestClassifier(n_estimators=24, random_state=0, max_depth=5)
rf_model.fit(samples_train, targets_train)
time_end = time.perf_counter()

print("Random Forest Model Results\nThe average accuracy of Random Forest classification with N_SAMPLES=" + str(n) + " and noise = " + str(noise) +  "% is: " + str(100*rf_model.score(samples_test, targets_test)) + "%.\nTraining completed in " + str(time_end - time_start) + " seconds.\n")

print("Beginning KNeighbors Classification...\n")
from sklearn.neighbors import KNeighborsClassifier
time_start = time.perf_counter()
knn_model = KNeighborsClassifier()
knn_model.fit(samples_train, targets_train)
time_end = time.perf_counter()

print("K-nearest neighbors model results\nThe average accuracy of KNN classification with N_SAMPLES=" + str(n) + " and noise = " + str(noise) +  "% is: " + str(100*knn_model.score(samples_test, targets_test)) + "%.\nTraining completed in " + str(time_end - time_start) + " seconds.\n")


#Make an array that contains 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
#n times

svm_target_array = []
for i in range(n*10):
    #every 10th element is the letter 'A'
    if i % 10 == 0:
        svm_target_array.append('A')
    #every 11th element is the letter 'B'
    elif i % 10 == 1:
        svm_target_array.append('B')
    #every 12th element is the letter 'C'
    elif i % 10 == 2:
        svm_target_array.append('C')
    #every 13th element is the letter 'D'
    elif i % 10 == 3:
        svm_target_array.append('D')
    #every 14th element is the letter 'E'
    elif i % 10 == 4:
        svm_target_array.append('E')
    #every 15th element is the letter 'F'
    elif i % 10 == 5:
        svm_target_array.append('F')
    #every 16th element is the letter 'G'
    elif i % 10 == 6:
        svm_target_array.append('G')
    #every 17th element is the letter 'H'
    elif i % 10 == 7:
        svm_target_array.append('H')
    #every 18th element is the letter 'I'
    elif i % 10 == 8:
        svm_target_array.append('I')
    #every 19th element is the letter 'J'
    elif i % 10 == 9:
        svm_target_array.append('J')
    
    

samples_train, samples_test, svm_targets_train, svm_targets_test = train_test_split(samples.T, svm_target_array, test_size=0.40, random_state=1)


#use svm to classify the data
print("Beginning SVM classification...\n")
from sklearn.svm import SVC
time_start = time.perf_counter()
svm_model = SVC()
svm_model.fit(samples_train, svm_targets_train)
time_end = time.perf_counter()

print("SVM Model Results\nThe average accuracy of SVM classification with N_SAMPLES=" + str(n) + " and noise = " + str(noise) +  "% is: " + str(100*svm_model.score(samples_test, svm_targets_test)) + "%.\nTraining completed in " + str(time_end - time_start) + " seconds.\n")




# %%
#Now we will create some bar graphs comparing the accuracy of the different classifiers
import matplotlib.pyplot as plt

#create a bar graph that compares categories
categories = ['n = 5, noise = 10%', 'n = 5, noise = 15%', 'n = 5, noise = 25%', 'n = 15, noise = 10%', 'n = 15, noise = 15%', 'n = 15, noise = 25%', 'n = 45, noise = 10%', 'n = 45, noise = 15%', 'n = 45, noise = 25%']

mlp_scores = [35.0, 40, 40.0, 20.0, 20.0, 20.0, 25.0, 15.0, 25.0]
rf_scores = [30.0, 35.0, 25.0, 35.0, 25.0, 25.0, 35.0, 40.0, 25.0]
knn_scores = [20.0, 35.0, 20.0, 35.0, 30.0, 20.0, 25.0, 25.0, 25.0]
svm_scores = [0, 0, 0, 3.33, 3.33, 5.0, 3.89, 2.77, 3.33]

#plot mlp_scores scores vs categories
plt.bar(categories, mlp_scores, color='b', align='center')
plt.title('MLP Model Accuracy at Different N_SAMPLES and NOISE')
plt.ylabel('Accuracy (%)')
plt.xlabel('N_SAMPLES and NOISE')
#make the x-axis labels readable
plt.xticks(categories, rotation=90, fontsize=8, fontweight='bold')
#shift the x-axis labels to the left
plt.show()

#now doing the same for rf_scores
plt.bar(categories, rf_scores, color='r', align='center')
plt.title('Random Forest Model Accuracy at Different N_SAMPLES and NOISE')
plt.ylabel('Accuracy (%)')
plt.xlabel('N_SAMPLES and NOISE')
#make the x-axis labels readable
plt.xticks(categories, rotation=90, fontsize=8, fontweight='bold')
#shift the x-axis labels to the left
plt.show()

#now doing the same for knn_scores
plt.bar(categories, knn_scores, color='g', align='center')
plt.title('KNN Model Accuracy at Different N_SAMPLES and NOISE')
plt.ylabel('Accuracy (%)')
plt.xlabel('N_SAMPLES and NOISE')
#make the x-axis labels readable
plt.xticks(categories, rotation=90, fontsize=8, fontweight='bold')
#shift the x-axis labels to the left
plt.show()

#now doing the same for svm_scores
plt.bar(categories, svm_scores, color='y', align='center')
plt.title('SVM Model Accuracy at Different N_SAMPLES and NOISE')
plt.ylabel('Accuracy (%)')
plt.xlabel('N_SAMPLES and NOISE')
#make the x-axis labels readable
plt.xticks(categories, rotation=90, fontsize=8, fontweight='bold')
#shift the x-axis labels to the left
plt.show()

# %%
