import pandas as pd

class Frame3D:
    def __init__(self, df: pd.DataFrame):
        assert isinstance(df.index, pd.MultiIndex), "Index must be MultiIndex (key, name)"
        self._df = df

    @property
    def df(self):
        return self._df

    def copy(self):
        return Frame3D(self._df.copy(deep=True))

    def __repr__(self):
        return f"Frame3D({repr(self._df)})"
