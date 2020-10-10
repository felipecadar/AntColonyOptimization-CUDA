import random
a = [0.1, 0.1, 0, 0.4, 0.4]
b = [0,0,0,0,0]


for i in range(100000):
    r = random.random()
    cum = 0
    for j in range(5):
        if r < a[j] + cum:
            b[j] += 1
            break
        cum += a[j]


print(b)