# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:29:54 2021

@author: Gabri
"""

from Functions import Utilities
import pandas as pd

example_frts=pd.read_excel("Examples/Consumo_Mes_dia_TxR_aenc01.xlsx")
test=Utilities.check_dtype_data(example_frts)