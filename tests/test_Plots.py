import unittest
import pandas as pd
import numpy as np
from plots import remove_top_percentage

class TestRemoveTop(unittest.TestCase):

    def setUp(self):
        n = 20
        dfs = []
        for budget in range(5):
            df = pd.DataFrame({
                "budget": np.repeat(budget, n),
                "mae": (1+budget)*np.random.normal(1, 1, n)
            })
            dfs.append(df)

        self.df = pd.concat(dfs)

    def test_(self):
        df = self.df.copy()
        df_ = remove_top_percentage(df, 0.05, ["mae"], df["budget"])
        df__ = remove_top_percentage(df, 0.20, ["mae"], df["budget"])

        self.assertEqual(0.95*len(df), len(df_))
        self.assertEqual(0.8*len(df), len(df__))

        self.assertTrue(df["mae"].sum() > df_["mae"].sum())
        self.assertTrue(df_["mae"].sum() > df__["mae"].sum())

        cutoff = df[df["budget"]==3]["mae"].quantile(q=0.79)

        x1 = df[(df["budget"]==3)&(df["mae"]<cutoff)]["mae"].sum()
        x2 = df_[(df_["budget"]==3)&(df_["mae"]<cutoff)]["mae"].sum()
        x3 = df__[(df__["budget"]==3)&(df__["mae"]<cutoff)]["mae"].sum()
        self.assertAlmostEqual(x1, x2)
        self.assertAlmostEqual(x2, x3)

