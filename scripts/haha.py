import random

i = 0
# Open the input file in read mode
with open('points3D.txt','w') as new:
    with open('output.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if i%10 == 0:
                new.write(''.join(line))
            i=i+1
