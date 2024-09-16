from preprocess import PreProcess, Select

from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OrdinalEncoder
)
from typing import (
    List,
    Union,
)

class OrdinalEncoding(nn.Module):
    def __init__(
            self,
            dtype=np.int64,
            *args,
            **kwargs) -> None:
        """
        Transform text into an integer category.

        Args:
            dtype: Desired dtype of replacement values
        """
        super().__init__(*args, **kwargs)
        self.dtype = dtype
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        mask = x.dtypes == 'object'
        oe = OrdinalEncoder(dtype=self.dtype, encoded_missing_value=-1).fit_transform(x.loc[:,mask])
        x.loc[:,mask] = oe
        x[mask.index[mask]] = x[mask.index[mask]].astype(self.dtype)
        return x
    
class FillNaN(nn.Module):
    def __init__(
            self,
            selections:list[str] = None,
            fill_value:Union[int, List[int]] = -1,
            *args,
            **kwargs) -> None:
        """
        Transform text into an integer category.

        Args:
            selections: Specify what labels should be included.
            fill_value: Value with which to replace NaNs. Either a list of
              values the same size as selections, where each value will fill
              respectively, or a single value to be applied for all selections.
        """
        super().__init__(*args, **kwargs)
        self.selections = selections if selections else []
        self.fill_value = fill_value
        if not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value]*len(self.selections)
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        for i, selection in enumerate(self.selections):
            mask = x.loc[:,selection].isnull()
            x.loc[mask,selection] = self.fill_value[i]
        return x
    
class PPLabels(PreProcess):
    def __init__(
        self,
        selections = None,
        exclusions = None,
        exclude_uninformative = True,
        ordinal_encoding:bool = True,
        fill_nan_selections = None,
        fill_nan_values = None,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.select = Select(selections=selections, exclusions=exclusions, exclude_uninformative=exclude_uninformative)
        self.one_hot = OrdinalEncoding() if ordinal_encoding else None
        fill_nan_config = {'selections':fill_nan_selections}
        if fill_nan_values:
            fill_nan_config['fill_value'] = fill_nan_values
        self.fill_nan = FillNaN(**fill_nan_config)