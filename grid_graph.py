import pickle
import torch
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 파일 불러오기
with open("action.pkl", "rb") as f:
    action = pickle.load(f)

action = action.view(5, -1)
print(action)


# # Generate random data
# data = action

# # Create a heatmap
# plt.imshow(data, cmap="viridis", aspect="auto")
# plt.colorbar()

# # Show the plot
# plt.show()

# print(action[0])
# print(action[0][0])

# print(action[0].tolist())
# print(float(action[0][0]))


""" Divied and Padding """
################################################
final = []
pad = []
for i in range((13+2)*10):
    pad.append(0)

for i in range(10):
    final.append(pad)

for i in range(5):
    tmp = []
    for k in range(10):
        tmp.append(0)

    for j in range(13):
        for k in range(10):
            tmp.append(float(action[i][j]))

    for k in range(10):
        tmp.append(0)

    for k in range(10):
        final.append(tmp)

for i in range(10):
    final.append(pad)
################################################

# print(final)
data = torch.tensor(final)
print(data.shape)

# # Create a heatmap
# plt.imshow(data, cmap="viridis", aspect="auto")
# plt.colorbar()

# # Show the plot
# plt.show()


""" Average pooling """
size = 19

def average_pooling(x, y):
    tmp = 0

    for j in range(y, y+size):
        for i in range(x, x+size):
            # print((x, y))
            # print((i, j))
            tmp += data[i][j]

    return tmp/(size**2)



final2 = []
for i in range(50+20-size+1+1-1):
    tmp2 = []
    for j in range(130+20-size+1):
        # print('1111111111')
        # print(i)
        mean_value = average_pooling(i, j)
        tmp2.append(mean_value)

    final2.append(tmp2)

data2 = torch.tensor(final2)
print(data2.shape)

# # Create a heatmap
# plt.imshow(data2, cmap="viridis", aspect="auto")
# plt.colorbar()

# # Show the plot
# plt.show()


final3 = []

for i in range(50):
    tmp3 = []

    for j in range(128):
        tmp3.append(float(data2[i][j]))

    final3.append(tmp3)


data3 = torch.tensor(final3)
print(data3.shape)

with open("RLPE.pkl", "wb") as f:
    pickle.dump(data3, f)

# Create a heatmap
plt.imshow(data3, cmap="viridis", aspect="auto")
plt.colorbar()

# Show the plot
plt.show()
