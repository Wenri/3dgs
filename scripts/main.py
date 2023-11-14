# Open the input file in read mode
with open('fused.txt', 'r') as file:
    lines = file.readlines()

# Add a new column to each line of the file
modified_lines = [line.strip() + ' 0.7' for line in lines]

# Open the output file in write mode
with open('points3D.txt', 'w') as file:
    file.write('\n'.join(modified_lines))
