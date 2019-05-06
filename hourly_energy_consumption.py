



import pandas as pd



def x_y_p(filepath):
    csv=pd.read_csv(filepath)
    data=[]
    for col in csv:
        temp=list()
        for x in csv[col]:
            temp.append(x)
        data.append(temp)
    temp=data[1]
    data=temp
    if filepath=="hourly-energy-consumption/FE_hourly.csv":
        data[5111]=data[5110]

    hours_in_a_week=7*24
    total_weeks=len(data)//(hours_in_a_week)

    x_axis=[]
    y_axis=[]
    period=[]

    for i in range(total_weeks-7+1): # all 7 weeks
        pivot_index=i*hours_in_a_week
        for j in range(3,8): # period length
            period_length=j*hours_in_a_week
            for k in range(7-j+1): #pivot of subweek in 7 weeks
                pivot=k*hours_in_a_week
                period.append(j)
                temp=[]
                for num in data[pivot_index+pivot:pivot_index+pivot+period_length]:
                    temp.append(num)
                y_axis.append(temp)
                temp=[]
                for l in range(len(data[pivot_index+pivot:pivot_index+pivot+period_length])):
                    temp.append(l)
                x_axis.append(temp)
    return x_axis,y_axis,period 


def make_2d_plot():
    names=["hourly-energy-consumption/AEP_hourly.csv","hourly-energy-consumption/COMED_hourly.csv","hourly-energy-consumption/DAYTON_hourly.csv","hourly-energy-consumption/DEOK_hourly.csv","hourly-energy-consumption/DOM_hourly.csv","hourly-energy-consumption/DUQ_hourly.csv","hourly-energy-consumption/EKPC_hourly.csv","hourly-energy-consumption/FE_hourly.csv","hourly-energy-consumption/NI_hourly.csv","hourly-energy-consumption/PJM_Load_hourly.csv","hourly-energy-consumption/PJME_hourly.csv","hourly-energy-consumption/PJMW_hourly.csv"]
    x_axis=[]
    y_axis=[]
    period=[]
    for name in names:
        temp=x_y_p(name)
        x_axis+=temp[0]
        y_axis+=temp[1]
        period+=temp[2]

    return x_axis,y_axis,period









