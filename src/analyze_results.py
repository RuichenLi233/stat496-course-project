import pandas as pd


df = pd.read_csv("results.csv")

# 1. Overall accuracy
overall_acc = df["is_exact_match"].mean()
print("Overall accuracy:")
print(overall_acc)
print()


acc_by_temp = df.groupby("temperature")["is_exact_match"].mean()
print("Accuracy by temperature:")
print(acc_by_temp)
print()


group_cols = ["qid", "temperature", "top_p", "style"]
diversity = df.groupby(group_cols)["response"].nunique().mean()
print("Average diversity per setting:")
print(diversity)
print()


grouped = df.groupby(group_cols)["response"].nunique()
stability = (grouped == 1).mean()
print("Stability rate:")
print(stability)
