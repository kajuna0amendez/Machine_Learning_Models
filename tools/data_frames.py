# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2018"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Apache"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import pandas as pd
import numpy as np

def dframe_t_db(rows, name):
    """
    Assambly all the information from a database
    """
    # Zip through the indexes
    cols = np.vstack(rows)

    print(cols.shape)
    print(len(name))

    return pd.DataFrame(cols, columns = name)
