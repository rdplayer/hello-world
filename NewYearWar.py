import numpy as np
import random

x=np.zeros(100)
num = 0
max = 0
win = True
for i in range(10):
    for k in range(100):
        n = random.randint(1,4)
        n1 = random.randint(1,4)
        if n1 == n :
            if win:
                num += 1
                if num > max :
                    max = num
                win = True
            else:
                win = True
                num = 1
        else:
            win = False
            num = 0

    print(max)
    max = 0
    num = 0