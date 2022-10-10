import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import calplot
import numpy as np
from calendar import monthrange
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc # For the legend
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, date2num, DateFormatter, HourLocator, MinuteLocator
from sqlalchemy import create_engine
import json

def executeCommand(cursor, command):
    """
    Use to excecute MySQL command.

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    command : string
        A string which is MySQL query to send the command to a MySQL server.

    Return
    ---------------
    result : string
        If command is able to be excuted, the function will print 'Command excuted - ' after with the query.
        If command is fail, the function will print 'Command fail - ' after with an error.
    """
    try:
        #cursor.execute(command)
        cursor.execute(command)
        result = 'Command excuted - {}'.format(command)
    except Exception as e:
        result = 'Command fail - {}'.format(e)
    finally:
        return result


def createDatabase(cursor, database_name):
    """
    Use to create a database to a MySQL server

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    database_name : string
        MySQL database name.

    Return
    ---------------
        Execute the command and return a string from executeCommand function.
    """
    # Create Database
    command = 'CREATE DATABASE {database_name};'.format(database_name=database_name)
    return executeCommand(cursor, command)


def useDatabase(cursor, database_name):
    """
    Use to determine whata database from MySQL server to be used.

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    database_name : string
        MySQL database name.

    Return
    ---------------
        Execute the command and return a string from executeCommand function.
    """
    # Use Database
    command = 'USE {}'.format(database_name)
    return executeCommand(cursor, command)


def createTableGeneration(cursor, columns_names, table_name):
    """
    Create a generation table in MySQL server.

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    columns_names : string
        MySQL column name.

    table_name : string
        MySQL table name.

    Return
    ---------------
        Execute the command and return a string from executeCommand function.
    """
    # Create table for Generation_Data
    columns_names_iter = iter(columns_names)

    command = """CREATE TABLE {} (
        {} TIMESTAMP UNIQUE,
        {} DECIMAL(15,5),
        {} INT,
        {} INT,
        {} INT,
        {} INT,
        {} INT,
        {} INT
        )""".format(table_name,
                    next(columns_names_iter), next(columns_names_iter), next(columns_names_iter),
                    next(columns_names_iter), next(columns_names_iter), next(columns_names_iter),
                    next(columns_names_iter), next(columns_names_iter))
    return executeCommand(cursor, command)


def delAllRow(cursor, table_name):
    """
    Use to determine whata database from MySQL server to be used.

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    table_name : string
        MySQL table name.

    Return
    ---------------
        Execute the command and return a string from executeCommand function.
    """
    # Use Database
    command = 'DELETE FROM {};'.format(table_name)
    return executeCommand(cursor, command)


def importData(cursor, file_path, table_name):
    """
    Import data from a file to a table.

    Inputs
    ---------------
    cursor : mysql.connector.cursor.MySQLCursor
        MySQL curror

    file_path : string
        A file location.

    table_name : string
        MySQL column name.

    Return
    ---------------
        Execute the command and return a string from createTableGeneration function.
    """
    command = """
    LOAD DATA INFILE '{}'
    INTO TABLE {}
    FIELDS TERMINATED BY ','
    ENCLOSED BY '"'
    LINES TERMINATED BY '\n'
    IGNORE 1 ROWS;
    """.format(file_path, table_name)
    return executeCommand(cursor, command)


def seperateDatetime(df, date_namecolumn='Unnamed: 0'):
    df_temp = df.copy()
    df_temp[date_namecolumn] = pd.to_datetime(df_temp[date_namecolumn])
    df_temp['year'] = df_temp[date_namecolumn].dt.year
    df_temp['month'] = df_temp[date_namecolumn].dt.month
    df_temp['day'] = df_temp[date_namecolumn].dt.day
    df_temp['hour'] = df_temp[date_namecolumn].dt.hour
    df_temp['minute'] = df_temp[date_namecolumn].dt.minute
    df_temp['second'] = df_temp[date_namecolumn].dt.second
    #df.drop(columns='Unnamed: 0', inplace=True)
    #df_temp.head()

    return df_temp


def dateFilter(df, date_filter, date_namecolumn='Unnamed: 0'):
    df_temp = df.copy()
    if date_filter['year'] is not None:
        df_temp = df_temp[df_temp['year']==date_filter['year']]
        #display(df.head())
    if date_filter['month'] is not None:
        df_temp = df_temp[df_temp['month']==date_filter['month']]
        #display(df.head())
    if date_filter['day'] is not None:
        df_temp = df_temp[df_temp['day']==date_filter['day']]
        #display(df.head())
    if date_filter['hour'] is not None:
        df_temp = df_temp[df_temp['hour']==date_filter['hour']]
        #display(df.head())
    return df_temp


def pivotTable(df, date_filter, index='second', columns='minute'):
    df_temp = dateFilter(df, date_filter)

    df_pivot = df_temp.pivot_table(index=index, columns=columns, values='Frequency')
    return df_pivot


def reshapeToPlot(row_num, a_list):
    col_mod = int(len(a_list)%row_num)
    if col_mod != 0:
        col_num = int((len(a_list)-col_mod+row_num)/row_num)
    else:
        col_num = int((len(a_list))/row_num)

    a_list_reshape = np.array(a_list)
    a_list_reshape = np.append(a_list_reshape, list(range(0, (row_num*col_num) - len(a_list))))
    a_list_reshape = a_list_reshape.reshape(row_num, col_num)

    return a_list_reshape


def createTitle(title, date_filter):
    year = date_filter['year']
    month = date_filter['month']
    day = date_filter['day']
    if day is not None:
        title = title + str(day) + '.'
    if month is not None:
        title = title + str(month) + '.'
    if year is not None:
        title = title + str(year)
    return title


def plotHeatMap(df, date_filter, index, columns, cbar={'display':False, 'max':150, 'min':-150, 'auto':False}):
    df_temp = df.copy()
    if date_filter['pointer'] == 'day':
        month_number = monthrange(date_filter['year'], date_filter['month'])[1]
        pointers = list(range(1, month_number+1))
    else:
        pointers = list(df_temp[date_filter['pointer']].unique())
    title = createTitle('Heatmap for ', date_filter)

    fig, axes = plt.subplots(len(pointers), 1, figsize=(20, 15), sharex=True)
    #fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True)
    fig.suptitle(title, fontsize=18)

    gain = 10
    if cbar['auto'] is True:
        #print('cbar_auto = True')
        #vmax = df_temp['Frequency'].max() + gain
        #vmin = df_temp['Frequency'].min() - gain
        #print(vmax)
        #print(vmin)
        vmax = None
        vmin = None
    else:
        #print('cbar_auto = False')
        vmax = cbar['max']
        vmin = cbar['min']

    for i, pointer in enumerate(pointers):
        date_filter[date_filter['pointer']] = pointer
        df_pivot = pivotTable(df_temp, date_filter, index=index, columns=columns)
        #display(df_pivot)

        # Plot heat map
        if len(pointers)==1:
            sns.heatmap(df_pivot, vmin=vmin, vmax=vmax, cbar=cbar['display'], ax=axes)
            axes.set_ylabel('{}-{}'.format(date_filter['pointer'], pointer), rotation = 0)
            axes.yaxis.set_label_coords(-0.01*4,0.5)
        else:
            sns.heatmap(df_pivot, vmin=vmin, vmax=vmax, cbar=cbar['display'], ax=axes[i])
            axes[i].set_ylabel('{}-{}'.format(date_filter['pointer'], pointer), rotation = 0)
            axes[i].yaxis.set_label_coords(-0.01*4,0.5)


    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, hspace=0.08, wspace=0.04)

"""
def plotLine(df, date_filter, figsize=(80,15)):
    df_temp = dateFilter(df, date_filter)
    x = df_temp['Unnamed: 0']
    y = df_temp['Frequency']

    title = createTitle('Frequency for ', date_filter)

    fig, axe = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=60)

    axe.plot(x, y, linewidth=3, label=y.name)
    axe.axhline(y=y.mean(), color='r')
    axe.text(x.iloc[0], y.mean()+3, 'mean={}'.format(str(y.mean())), fontsize='25', color='r')

    min_date = min(x)
    max_date = max(x)
    min_date_num = date2num(min_date)
    max_date_num = date2num(max_date)
    # Set the x-axis minor tick marks
    axe.set_xlim([min_date_num, max_date_num])

    # Tick label format style
    #dateFmt = DateFormatter('%d.%m.%Y %H:%M:%S')
    dateFmt = DateFormatter('%d.%m.%Y')
    #dateFmt = DateFormatter('%d')

    # For X Locator
    locator = AutoDateLocator()

    # Set the x-axis major tick marks
    axe.xaxis.set_major_locator(locator)
    # Set the x-axis labels
    axe.xaxis.set_major_formatter(dateFmt)
    axe.tick_params(axis='x', labelsize=35, width=7, length=20, direction='inout')
    axe.tick_params(axis='y', labelsize=35, width=7, length=20, direction='inout')
    axe.set_xlabel("Date", fontsize=50)
    axe.set_ylabel("Frequency [mHz]", fontsize=50)
    #axe.grid(which='major', linestyle = '-', linewidth = 2)
    #axe.grid(which='minor', linestyle = '--', linewidth = 1)
    #axe.legend(fontsize=15, loc='upper left')

    plt.show()
"""

def plotLine(x, y, title, figsize=(80,15)):
    #df_temp = dateFilter(df, date_filter)
    #x = df_temp['Unnamed: 0']
    #y = df_temp['Frequency']

    #title = createTitle('Frequency for ', date_filter)

    fig, axe = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=60)

    axe.plot(x, y, linewidth=3, label=y.name)
    axe.axhline(y=y.mean(), color='r')
    axe.text(x.iloc[0], y.mean()+3, 'mean={}'.format(str(y.mean())), fontsize='25', color='r')

    min_date = min(x)
    max_date = max(x)
    min_date_num = date2num(min_date)
    max_date_num = date2num(max_date)
    # Set the x-axis minor tick marks
    axe.set_xlim([min_date_num, max_date_num])

    # Tick label format style
    #dateFmt = DateFormatter('%d.%m.%Y %H:%M:%S')
    dateFmt = DateFormatter('%d.%m.%Y')
    #dateFmt = DateFormatter('%d')

    # For X Locator
    locator = AutoDateLocator()

    # Set the x-axis major tick marks
    axe.xaxis.set_major_locator(locator)
    # Set the x-axis labels
    axe.xaxis.set_major_formatter(dateFmt)
    axe.tick_params(axis='x', labelsize=35, width=7, length=20, direction='inout')
    axe.tick_params(axis='y', labelsize=35, width=7, length=20, direction='inout')
    axe.set_xlabel("Date", fontsize=50)
    axe.set_ylabel("Frequency [mHz]", fontsize=50)
    #axe.grid(which='major', linestyle = '-', linewidth = 2)
    #axe.grid(which='minor', linestyle = '--', linewidth = 1)
    #axe.legend(fontsize=15, loc='upper left')

    plt.show()
    return axe


def readCSV_fromYear(year):
    months = list(range(1,13))
    path_name = 'dataset/OSF_Storage/Continental_Europe/'
    df_list = []
    for month in months:
        try:
            file_name = path_name + 'Germany/{year1}/{month1:02d}/germany_{year2}_{month2:02d}.csv.zip'.format(year1= year, month1=month, year2=year, month2=month)
            df_temp = pd.read_csv(file_name)
            df_list.append(df_temp)
        except:
            pass
    df = pd.concat(df_list)

    return df


def plotBoxplot(df, x='month', figsize=(80,15)):
    #df_temp = df.copy()

    title = 'Box plot for {}'.format(df['year'].unique()[0])

    fig, axe = plt.subplots(figsize=(80,15))
    fig.suptitle(title, fontsize=60)

    sns.boxplot(x=x, y="Frequency", data=df)
    #sns.stripplot(x=x, y="Frequency", data=df_temp, marker="o", alpha=0.3, color="black")

    axe.tick_params(axis='x', labelsize=35, width=7, length=20, direction='inout')
    axe.tick_params(axis='y', labelsize=35, width=7, length=20, direction='inout')
    axe.set_xlabel("{}".format(x), fontsize=50)
    axe.set_ylabel("Frequency [mHz]", fontsize=50)
