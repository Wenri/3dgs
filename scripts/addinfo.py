import json
import numpy as np

# Read the original JSON file
with open('camera.json', 'r') as json_file:
    original_data = json.load(json_file)

# Create updated_data to store the updated information
updated_data = []
i = 0

for data in original_data:
    #new_cx_add = data['cx'] 
    #new_cy_add = data['cy'] 
    #new_cx_sub = data['cx'] 
    #new_cy_sub = data['cy'] 
    
    position_data_aa = np.array(data['position'])+[1, 1, 0.0]
    position_data_as = np.array(data['position'])+[1, -1, 0.0]
    position_data_sa= np.array(data['position'])+[-1, 1, 0.0]
    position_data_ss= np.array(data['position'])+[-1, -1, 0.0]
    rotation_data = np.array(data['rotation'])

    RT_aa = np.zeros((4,4))
    RT_aa[:3,:3] = rotation_data
    RT_aa[:3, 3] = position_data_aa
    RT_aa[3,3] = 1.0
    C2W_aa = np.linalg.inv(RT_aa)
    R_aa = C2W_aa[:3,:3].transpose().tolist()
    T_aa = C2W_aa[:3,3].tolist()

    RT_as = np.zeros((4,4))
    RT_as[:3,:3] = rotation_data
    RT_as[:3, 3] = position_data_as
    RT_as[3,3] = 1.0
    C2W_as = np.linalg.inv(RT_as)
    R_as = C2W_as[:3,:3].transpose().tolist()
    T_as = C2W_as[:3,3].tolist()

    RT_sa = np.zeros((4,4))
    RT_sa[:3,:3] = rotation_data
    RT_sa[:3, 3] = position_data_sa
    RT_sa[3,3] = 1.0
    C2W_sa = np.linalg.inv(RT_sa)
    R_sa = C2W_sa[:3,:3].transpose().tolist()
    T_sa = C2W_sa[:3,3].tolist()

    RT_ss = np.zeros((4,4))
    RT_ss[:3,:3] = rotation_data
    RT_ss[:3, 3] = position_data_ss
    RT_ss[3,3] = 1.0
    C2W_ss = np.linalg.inv(RT_ss)
    R_ss = C2W_ss[:3,:3].transpose().tolist()
    T_ss = C2W_ss[:3,3].tolist()





    # Create the new information
    new_info_1 = {
        "id": data['id']+i,
        "img_name": data['img_name']+'_1',
        "width": data['width'],
        "height": data['height'],
        "position": position_data_aa.tolist(),
        "rotation": data['rotation'],
        "fy": data['fy'],
        "fx": data['fx'],
        "R": R_aa,
        "T": T_aa,
        #"cx": new_cx_add,
        #"cy": new_cy_add
    }

    new_info_2 = {
        "id": data['id']+i+1,
        "img_name": data['img_name']+'_2',
        "width": data['width'],
        "height": data['height'],
        "position": position_data_as.tolist(),
        "rotation": data['rotation'],
        "fy": data['fy'],
        "fx": data['fx'],
        "R": R_as,
        "T": T_as,
        #"cx": new_cx_add,
        #"cy": new_cy_sub
    }

    new_info_3 = {
        "id": data['id']+i+2,
        "img_name": data['img_name']+'_3',
        "width": data['width'],
        "height": data['height'],
        "position": position_data_sa.tolist(),
        "rotation": data['rotation'],
        "fy": data['fy'],
        "fx": data['fx'],
        "R": R_sa,
        "T": T_sa,
        #"cx": new_cx_sub,
        #"cy": new_cy_add
    }

    new_info_4 = {
        "id": data['id']+i+3,
        "img_name": data['img_name']+'_4',
        "width": data['width'],
        "height": data['height'],
        "position": position_data_ss.tolist(),
        "rotation": data['rotation'],
        "fy": data['fy'],
        "fx": data['fx'],
        "R": R_ss,
        "T": T_ss,
        #"cx": new_cx_sub,
        #"cy": new_cy_sub
    }

    data['id'] = data['id']+i+4
    i = i+4
    # Add the new information to updated_data
    updated_data.append(new_info_1)
    updated_data.append(new_info_2)
    updated_data.append(new_info_3)
    updated_data.append(new_info_4)
    updated_data.append(data)

# Write the updated data back to the JSON file
with open('camera.json', 'w') as json_file:
    json.dump(updated_data, json_file)

print("New information has been created and the JSON file has been updated.")
