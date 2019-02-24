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

from data_model.load_data import create_connection, select_all_tasks
from tools.data_frames import dframe_t_db

def main():
    database = "/Cython_Code/database/heart.db"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("2. Query all tasks")
        rows, name = select_all_tasks(conn, 'heart_table')
    return dframe_t_db(rows, name)
 
if __name__ == '__main__':
    df = main()

    print(df)



