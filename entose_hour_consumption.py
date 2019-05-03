



import math
import pandas as pd

def consecutive_nan_zero(data,index):
    temp=0
    i=index
    consecutive=True
    while(consecutive):
        if math.isnan(data[i]) or data[i]==0:
            temp+=1
        else:
            consecutive=False
        if(i+1==len(data)):
            consecutive=False
        else:
            i+=1
    return temp

def twod_data(f,t,data):
    final=[]
    x=[]
    y=[]
    p=[]
    max_period=(t-f+1)//365
    for hour in range(len(data)):
        for i in range(max_period):
            for j in range(i+1,max_period+1):
                p.append(j-i)
                temp=[]
                for num in range(f+i*365,f+j*365):
                    temp.append(num)
                x.append(temp)
                temp=[]
                for num in data[hour][f+i*365:f+j*365]:
                    temp.append(num)
                y.append(temp)
    final.append(x)
    final.append(y)
    final.append(p)
    return final



def make_2d_plot():
    csv=pd.read_csv("Monthly-hourly-load-values_2006-2015.csv")
    data=[]
    for col in csv:
        temp=list()
        for x in csv[col]:
            temp.append(x)
        data.append(temp)

    for i in range(3):
        del data[1]

    countries=dict()
    for country in data[0]:
        if country in countries:
            countries[country]+=1
        else:
            countries[country]=1

    del data[0]


    #################
    #NaN/zero 
    ################
    for i in range(len(data)):
        j=0
        temp=[]
        while(j!=len(data[i])):
            if math.isnan(data[i][j]) or data[i][j]==0:
                run=consecutive_nan_zero(data[i],j)
                for k in range(run):
                    temp.append(data[i][j-1])
                j+=run
            else:
                temp.append(data[i][j])
                j+=1
        data[i]=temp



    index=[]
    for name in countries:
        index.append(countries[name])
    del countries

    index2=[]
    for i in range(len(index)):
        index2.append(sum(index[:i+1]))

    temp=[]
    for i in range(len(index)):
        temp1=[]
        if i==0:
            temp1.append(0)
            temp1.append(index[0]-(index[0]%365)-1)
        else:
            temp1.append(index2[i-1])
            temp1.append(temp1[0]+index[i]-(index[i]%365)-1)
        temp.append(temp1)
    index=temp
    del index2


    temp=twod_data(index[0][0],index[0][1],data)
    x_axis=temp[0]
    y_axis=temp[1]
    period=temp[2]
    for i in range(1,len(index)):
        temp=twod_data(index[i][0],index[i][1],data)
        x_axis+=temp[0]
        y_axis+=temp[1]
        period+=temp[2]
        
    return x_axis,y_axis,period









