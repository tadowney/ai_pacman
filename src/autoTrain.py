import subprocess
import time
import numpy as np

episodes = 10000
for i in range(0,episodes):
    print("Running Episode", i)
    result = subprocess.run("python pacman.py -p ReflexAgent -k 2 --frameTime 0", stdout=subprocess.PIPE)
    test = str(result.stdout)
    winFlag = 'Lose'
    if (test.find('victorious') != -1): 
        winFlag = 'Win' 
    weights = np.loadtxt("weights.csv", delimiter=",")
    with open('oldWeights.txt', 'a') as f:
        f.write(" ".join(map(str, weights)))
        f.write(' ' + winFlag)
        f.write('\n')