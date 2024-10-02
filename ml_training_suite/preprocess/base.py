from torch import nn
import pandas as pd

from typing import Union

class PreProcess(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for proc in self._modules.values():
            x = proc(x)
        return x
    
class Select(nn.Module):
    def __init__(
            self,
            selections:list[str] = None,
            exclusions:list[str] = None,
            do_not_include:bool = False,
            exclude_uninformative:bool = True,
            *args,
            **kwargs) -> None:
        """
        Filter what is returned.

        Selections and exclusions are mutually exclusive options, only one
        should be used. Currently, selections and exclusions work only for
        pandas dataframes. Do_not_include will omit the entire object (any
        object) and return None instead, for purposes of saving memory.

        Args:
            selections: Specify what labels should be included.
            exclusions: Specify what labels should not be included.
            do_not_include: Omit entire input and instead return None.
            exclude_uninformative: Removes catagories for which all data is the 
              same or NaN.
        """
        super().__init__(*args, **kwargs)
        self.selections = selections
        self.exclusions = exclusions
        assert self.selections is None or self.exclusions is None
        self.do_not_include = do_not_include
        self.exclude_uninformative = exclude_uninformative
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        if self.do_not_include:
            return None
        else:
            sel = self.selections if self.selections else x._info_axis
            if self.exclusions:
                sel = sel.drop(self.exclusions)
            x = x[sel]
            if self.exclude_uninformative:
                uninf_mask = (x.loc[0] == x).all()
                x = x.drop(uninf_mask.index[uninf_mask], axis=1)
                x = x.dropna(axis=1, how='all')
            return x