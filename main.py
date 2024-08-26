import random
import numpy as np

d = False

config = {
    "dimensions": (3,3), 
    "generations": 5
}


tft_defect = 0

strategiesmem1 = {
    ## start, CC, CD, DC, DD
    "tit-for-tat": [1, 1, tft_defect, 1, tft_defect]
}

strategiesreactive = {
    ## start, C, D
    "tit-for-tat": [1, 1, tft_defect]
}

def organism_response_matrix(env_matrix, position, strategy, anger=True, start=False): ## forced at 2D only for now
    if anger:
        if start:
            return strategy[0]
        shape = env_matrix.shape
    #    cff = -1
    #    dimnoatperif = []
    #    for dimension in shape:
    ##        cff+=1
    #        if position[cff] == dimension-1 or position[cff] == 0:
    #            dimnoatperif.append()
        if d: print("angry")
        defiance = False
        y = position[0]
        x = position[1]
        loop_position = [-1,-1]
        for row in env_matrix:
            loop_position[0] += 1
            loop_position[1] = -1
            for organism in row:
                loop_position[1] +=1
                if d: print("in function: ", loop_position, len(row))
                if loop_position[0] == y or loop_position[0] == y-1 or loop_position[0] == y+1:
                    if loop_position[1] == x or loop_position[1] == x-1 or loop_position[1] == x+1:
                        if env_matrix[loop_position[0]][loop_position[1]] != 1:
                            defiance = True
                            if d: print("defiance")

        if defiance:
            return strategy[2]
        else:
            if d: print ("cooperation")
            return strategy[1]
        

#env_matrix = np.random.random(config["dimensions"])
env_matrix = np.array([[1, 1, 1], [1, 1, 1], [1,1, 0]])
print(env_matrix)


generational_matrix = []
generational_matrix.append(env_matrix.tolist())

current_env_matrix = env_matrix
next_env_matrix = []



for i in range(config["generations"]):
    position = [-1,-1]
    for row in current_env_matrix:
        position[0] += 1
        position[1] = -1
        rowmatrix = []
        for organism in row:
            position[1] += 1
            if d: print("in loop: ",position )
            rowmatrix.append(organism_response_matrix(current_env_matrix, position, strategiesreactive["tit-for-tat"]))
        
        next_env_matrix.append(rowmatrix)
    
    print(np.array(next_env_matrix))
    generational_matrix.append(next_env_matrix)
    current_env_matrix = np.array(next_env_matrix)
    next_env_matrix = []


print(generational_matrix)


import numpy as np
import matplotlib.pyplot as plt

x = [1, 1]
plt.plot(x)
plt.show()

fig, ax = plt.subplots()

min_val, max_val = 0, 15

intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in range(15):
    for j in range(15):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')

import numpy as np
import seaborn as sns
import pandas as pd

min_val, max_val = 0, 15
intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
cm = sns.light_palette("blue", as_cmap=True)
x=pd.DataFrame(intersection_matrix)
x=x.style.background_gradient(cmap=cm)
display(x)