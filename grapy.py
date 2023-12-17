import pickle
from matplotlib import pyplot as plt


# 파일 불러오기
with open("Rewards.pkl", "rb") as f:
    Rewards = pickle.load(f)

print(Rewards)
plt.plot(Rewards)
plt.show()
