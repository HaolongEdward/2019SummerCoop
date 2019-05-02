



import pandas as pd
import math
import matplotlib.pyplot as plt

def consecutive_nan(data,index):
    temp=0
    i=index
    consecutive=True
    while(consecutive):
        if math.isnan(data[i]):
            temp+=1
        else:
            consecutive=False
        if(i+1==len(data)):
            consecutive=False
        else:
            i+=1
    return temp
def make_2d_plot():
    csv=pd.read_csv("weather_madrid_LEMD_1997_2015.csv")
    temp_data=[]
    columns=[]

    for col in csv:
        columns.append(col)
        temp=list()
        for x in csv[col]:
            temp.append(x)
        temp_data.append(temp)

    #deleting non-periodical
    data=[]
    data.append(temp_data[1])
    data.append(temp_data[2])
    data.append(temp_data[3])
    data.append(temp_data[4])
    data.append(temp_data[5])
    data.append(temp_data[6])
    data.append(temp_data[10])
    data.append(temp_data[11])
    data.append(temp_data[12])

    #NaN
    max_nan_run=6
    for i in range(len(data)):
        j=0
        temp=[]
        while(j!=len(data[i])):
            if math.isnan(data[i][j]):
                run=consecutive_nan(data[i],j)
                if run>6:
                    j+=run
                else:
                    for k in range(run):
                        temp.append(data[i][j-1])
                    j+=run
            else:
                temp.append(data[i][j])
                j+=1
        data[i]=temp

    #x,y,period
    x_axis=[]
    y_axis=[]
    period=[]
    for i in range(len(data)): 
        max_period=len(data[i])//365
        for k in range(max_period):
            for l in range(k+1,max_period+1):
                temp=[]
                for m in range(l*365-k*365):
                    temp.append(m)
                x_axis.append(temp)
                y_axis.append(data[i][k*365:l*365])
                period.append(l-k)
    return x_axis,y_axis,period

    