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

import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    Param:
    -------------------------------------------------------------------
        db_file: database file
    
    Return:
    ------------------------------------------------------------------- 
        Connection object or None
    """
    # Try and catch in the case of error
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def select_all_tasks(conn, name):
    """
    Query all rows in the tasks table

    Param:
    -------------------------------------------------------------------
       conn: the Connection object
        
       name: string name for the table
    Return:
    ------------------------------------------------------------------- 
        
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM "+name)
    name = list(map(lambda x:x[0], cur.description))

    rows = list( map( lambda x: list(x), cur.fetchall()))

    return rows, name
