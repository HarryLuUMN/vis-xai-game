#%%
import torch
import numpy as np

#%%

def tranform_matrix_to_axis(adj_matrix):
    axis_a = []
    axis_b = []
    weights = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                axis_a.append(i)
                axis_b.append(j)
                weights.append(adj_matrix[i][j])
    return [axis_a, axis_b, weights]

# %%
import csv
import json
def csv_to_array(csv_path):
    array = []
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            array.append([int(element) for element in row])
    return array

# %%
test_path = r"DutchSchoolDataset\csv\net1_data.csv"
test_data = csv_to_array(test_path)
print(test_data)
# %%
test_axes = tranform_matrix_to_axis(test_data)
print(test_axes)

#%%

test_attr = csv_to_array(r"DutchSchoolDataset\csv\demo_data.csv")
print(test_attr)

#%%
advice = csv_to_array(r"DutchSchoolDataset\csv\advice_data.csv")
advice = [[element - 4 if element != 0 else element for element in row] for row in advice]
dict = {
    "x":test_attr,
    "edge_index":test_axes[:2],
    "y": advice
}

#%%
num_nodes = len(test_attr)
train_mask = [True if i < num_nodes * 0.6 else False for i in range(num_nodes)]
val_mask = [True if num_nodes * 0.6 <= i < num_nodes * 0.8 else False for i in range(num_nodes)]
test_mask = [True if i >= num_nodes * 0.8 else False for i in range(num_nodes)]

dict["train_mask"] = train_mask
dict["val_mask"] = val_mask
dict["test_mask"] = test_mask

#%%

def save_dict_to_json(data_dict, filename):
    with open(filename, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

save_dict_to_json(dict, r"DutchSchoolDataset\json\net1.json")



# %%
def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

test_axes = np.array(tranform_matrix_to_axis(test_data))
save_to_csv(test_axes.T.tolist(), r"DutchSchoolDataset\csv\net1_axes.csv")


# %%


