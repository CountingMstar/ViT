import pickle
import torch
from matplotlib import pyplot as plt


# 파일 불러오기
with open("action.pkl", "rb") as f:
    action = pickle.load(f)

action = action.view(5, -1)
print(action)


# Generate random data
data = action

# # Create a heatmap
# plt.imshow(data, cmap="viridis", aspect="auto")
# plt.colorbar()

# # Show the plot
# plt.show()

print(action[0])
print(action[0][0])

print(action[0].tolist())
print(float(action[0][0]))


""" Divied """
final = []
for i in range(5):
    tmp = []
    for j in range(13):
        for k in range(10):
            tmp.append(float(action[i][j]))
    # print("=========")
    # print(tmp)
    for k in range(10):
        final.append(tmp)

print(final)
data = torch.tensor(final)

# Create a heatmap
plt.imshow(data, cmap="viridis", aspect="auto")
plt.colorbar()

# Show the plot
plt.show()


""" Average pooling """
