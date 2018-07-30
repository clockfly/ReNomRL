import os
import sys
import numpy
import matplotlib.pyplot as plt

log_file_name = sys.argv[1]
if os.path.splitext(log_file_name)[-1] == "log":
    raise Exception()

train_reward = []
test_reward = []
train_loss = []
with open(log_file_name) as reader:
    for line in reader.readlines():
        one_line = line.strip()
        split = one_line.split(" ")
        train_loss.append(float(split[3]))
        train_reward.append(float(split[5]))
        test_reward.append(float(-1 if len(split) < 7 else split[7]))

plt.subplot("311")
plt.title('Loss')
plt.plot(train_loss)
plt.subplot("312")
plt.title('Train reward')
plt.plot(train_reward)
plt.subplot("313")
plt.title('Test reward')
plt.plot(test_reward)
plt.tight_layout()
plt.suptitle(log_file_name)
plt.subplots_adjust(top=0.9)
plt.show()
