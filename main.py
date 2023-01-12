import pandas as pd 
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("Covid Data.csv")
df = df_.copy()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

def association_(df, alive):
    df["ALIVE"] = df["DATE_DIED"].apply(lambda x: 1 if x == "9999-99-99" else 0)
    diseases = df[df["ALIVE"] == alive].iloc[:, 9:19]
    diseases = diseases.dropna()
    diseases = diseases.applymap(lambda x: True if x == 1 else False)
    supports = apriori(diseases, min_support = 0.0005, use_colnames = True)
    results = association_rules(supports, metric = "support", min_threshold = 0.0005)
    return results

deads = association_(df = df, alive = 0)

deads.groupby("antecedents")["support"].mean().sort_values(ascending = False)
