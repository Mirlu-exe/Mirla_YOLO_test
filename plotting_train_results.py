import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/nishiyamalab/PycharmProjects/Mirla_YOLO_test/runs/detect/train14/results.csv")
plt.plot(df["epoch"], df["metrics/mAP50(B)"])
plt.title("mAP50 over Epochs")
plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.show()

