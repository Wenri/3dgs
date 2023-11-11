import plyfile

# Path to the input PLY file
ply_file_path = 'fused.ply'

# Path to the output TXT file
txt_file_path = 'fused.txt'

# Load the PLY file
ply_data = plyfile.PlyData.read(ply_file_path)

# Open the output TXT file in write mode
with open(txt_file_path, 'w') as txt_file:
    # Iterate over each vertex in the PLY file
    for vertex in ply_data['vertex']:
        # Write the vertex coordinates to the TXT file
        txt_file.write(f"{vertex['x']} {vertex['y']} {vertex['z']} {vertex['red']} {vertex['green']} {vertex['blue']}\n")
