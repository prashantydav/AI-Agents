# from datasets import load_dataset
# ds = load_dataset("basicv8vc/SimpleQA")

import pandas as pd

df = pd.read_csv("hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv")

splits = {'eval': 'eval/reveal_eval.csv', 'open': 'open/reveal_open.csv'}
df_google = pd.read_csv("hf://datasets/google/reveal/" + splits["eval"])

df.to_csv("simple_qa_test_set.csv", index=False)

df_google.to_csv("reveal_eval.csv", index=False)