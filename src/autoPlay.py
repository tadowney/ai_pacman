import subprocess
import time
import numpy as np

episodes = 10000
for i in range(0,episodes):
    print("Running Episode", i)
    result = subprocess.run("python pacman.py -p ReflexAgent -l mediumClassic --frameTime 0", stdout=subprocess.PIPE)
    test = str(result.stdout)
    winFlag = 'Lose'
    if (test.find('victorious') != -1): 
        winFlag = 'Win' 
    with open('WinRatio.txt', 'a') as f:
        f.write(winFlag)
        f.write('\n')