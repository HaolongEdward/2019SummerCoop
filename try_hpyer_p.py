
import os
from server_CNN import CNN
import csv


def get_input():
    input_not_done=True
    print("must be same number of elements for each parameters")

    while(input_not_done):
        periods=input("periods: ")
        periods=eval("["+periods+"]")

        sizes=input("sizes: ")
        sizes=eval("["+sizes+"]")

        maxPolyDegrees=input("maxPolyDegrees: ")
        maxPolyDegrees=eval("["+maxPolyDegrees+"]")

        polyWeightVariances=input("polyWeightVariances: ")
        polyWeightVariances=eval("["+polyWeightVariances+"]")

        epochs=input("epochs: ")
        epochs=eval("["+epochs+"]")

        learning_rate=input("learning_rate: ")
        learning_rate=eval("["+learning_rate+"]")

        batch_size=input("batch_size: ")
        batch_size=eval("["+batch_size+"]")

        experiments_per_dataset=input("experiments_per_dataset: ")
        experiments_per_dataset=eval("["+experiments_per_dataset+"]")

        while(True):
            print("periods: ",periods)
            print("sizes: ",sizes)
            print("maxPolyDegrees: ",maxPolyDegrees)
            print("polyWeightVariances: ",polyWeightVariances)
            print("epochs: ",epochs)
            print("learning_rate: ",learning_rate)
            print("experiments_per_dataset: ",experiments_per_dataset)

            answer=input("enter \'y\' to run, enter \'n\' to re-enter, enter \'c\' to change one single parameters")
            while(True):
                if answer=="y" or answer=="n" or answer=="c":
                    break
                else:
                    answer=input("enter \'y\' to run, enter \'n\' to re-enter, enter \'c\' to change one single parameters")
            if answer=="y":
                input_not_done=False
                break
            elif answer=="n":
                break
            else:
                target=input(" 1-periods,\n 2-sizes,\n 3-maxPolyDegrees,\n 4-polyWeightVariances,\n 5-epochs,\n 6-learning_rate,\n 7-batch_size,\n 8-experiments_per_dataset")
                while(True):
                    if target in "12345678":
                        break
                    else:
                        target=input(" 1-periods,\n 2-sizes,\n 3-maxPolyDegrees,\n 4-polyWeightVariances,\n 5-epochs,\n 6-learning_rate,\n 7-batch_size,\n 8-experiments_per_dataset")
                if target=="1":
                    periods=input("periods: ")
                    periods=eval("["+periods+"]")
                if target=="2":
                    sizes=input("sizes: ")
                    sizes=eval("["+sizes+"]")
                if target=="3":
                    maxPolyDegrees=input("maxPolyDegrees: ")
                    maxPolyDegrees=eval("["+maxPolyDegrees+"]")
                if target=="4":
                    polyWeightVariances=input("polyWeightVariances: ")
                    polyWeightVariances=eval("["+polyWeightVariances+"]")
                if target=="5":
                    epochs=input("epochs: ")
                    epochs=eval("["+epochs+"]")
                if target=="6":
                    learning_rate=input("learning_rate: ")
                    learning_rate=eval("["+learning_rate+"]")
                if target=="7":
                    batch_size=input("batch_size: ")
                    batch_size=eval("["+batch_size+"]")
                if target=="8":
                    experiments_per_dataset=input("experiments_per_dataset: ")
                    experiments_per_dataset=eval("["+experiments_per_dataset+"]")
    return periods,sizes,maxPolyDegrees,polyWeightVariances,epochs,learning_rate,batch_size,experiments_per_dataset

def create_csv(filepath,directory):
    if len(directory)!=0:
        directory=directory+"/"
    file=open(filepath).read().splitlines()
    first_row=["epoch","loss","acc","val_loss","val_acc"]
    data=dict()
    temp=list()
    filenames=[]
    for i in range(len(file)):
        if "experiment:" in file[i].split():
            filenames.append(file[i+1].split()[2]+"_"+file[i].split()[2])
            # firstline : !!! experiment: xth .... 
            # second line: Experiment of XXX ...
            data[filenames[-1]]=list()
            if len(temp)!=0:
                data[filenames[-2]]=temp
                temp=list()
        if "loss:" in file[i].split():
            if "ETA:" in file[i].split():  # exclude the case "^M  32/2124 [..............................]....."
                pass
            else:
                numbers=[]
                for s in file[i].split():
                    try:
                        x=float(s)
                        numbers.append(x)
                    except:
                        pass
                temp.append(numbers)
    data[filenames[-1]]=temp
    for name in data:
        with open(directory+name+".csv","w") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(first_row)
            for i in range(len(data[name])):
                filewriter.writerow([i+1,]+data[name][i])
    #classifying, detecting number of experiments for each dataset, assuming they have equal number of experiments
    dataset_name=dict()
    for name in data:
        if name.split("_")[0] in dataset_name:
            dataset_name[name.split("_")[0]]+=1
        else:
            dataset_name[name.split("_")[0]]=1
    length=0
    for name in dataset_name:
        length=dataset_name[name]
        break
    #average
    dir_path=directory+"average_csv"
    os.mkdir(dir_path)
    for name in dataset_name:
        temp=[]
        for i in range(length):
            temp.append(data[name+"_"+str(i)+"th"])
        for i in range(1,length):
            for j in range(len(temp[i])):
                for k in range(len(temp[i][j])):
                    temp[0][j][k]+=temp[i][j][k]
        for i in range(len(temp[0])):
            for j in range(len(temp[0][i])):
                temp[0][i][j] = temp[0][i][j]/3
        with open(dir_path+"/"+name+".csv","w") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(first_row)
            for i in range(len(temp[0])):
                filewriter.writerow([i+1,]+temp[0][i])
    temp=[]#average csv names
    for name in dataset_name:
        temp.append(name)
    return temp


def create_latex_txt(names,directory):
    if len(directory)!=0:
        directory+="/"
    text_lines=list()
    text_lines.append("\\documentclass{amsart}\n")
    text_lines.append("\\usepackage{pgfplots}\n")
    text_lines.append("\\setlength{\\voffset}{-1cm}\n")
    text_lines.append("\\setlength{\\textwidth}{17cm}\n")
    text_lines.append("\\addtolength{\\textheight}{5cm}\n")
    text_lines.append("\\setlength{\\footskip}{1cm}\n")
    text_lines.append("\\addtolength{\\oddsidemargin}{-2cm}\n")
    text_lines.append("\\addtolength{\\evensidemargin}{-2cm}\n")
    text_lines.append("\\allowdisplaybreaks\n")
    text_lines.append("\\begin{document} \n")
    text_lines.append("\\thispagestyle{empty}~\\\\\n")
    text_lines.append("\\centerline{\\textbf{\\Huge "+ names[0]+"}}~\\vspace{0.8cm}\\\\")
    text_lines.append("\\begin{tikzpicture}\n")
    text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
    text_lines.append("\\addplot[color=blue,mark=*] table [x=epoch, y=loss, col sep=comma]{"+directory+names[0]+".csv};\n")
    text_lines.append("\\addlegendentry{loss}\n")
    text_lines.append("\\end{axis}\n")
    text_lines.append("\\end{tikzpicture}\\quad\\begin{tikzpicture}\n")
    text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
    text_lines.append("\\addplot[color=red,mark=*] table [x=epoch, y=val_loss, col sep=comma]{"+directory+names[0]+".csv};\n")
    text_lines.append("\\addlegendentry{\\texttt{val\\char`_loss}}\n")
    text_lines.append("\\end{axis}\n")
    text_lines.append("\\end{tikzpicture}\\\\")
    text_lines.append("\\begin{tikzpicture}\n")
    text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
    text_lines.append("\\addplot[color=teal,mark=*] table [x=epoch, y=acc, col sep=comma]{"+directory+names[0]+".csv};\n")
    text_lines.append("\\addlegendentry{acc}\n")
    text_lines.append("\\end{axis}\n")
    text_lines.append("\\end{tikzpicture}\\quad\\begin{tikzpicture}\n")
    text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
    text_lines.append("\\addplot[color=magenta,mark=*] table [x=epoch, y=val_acc, col sep=comma]{"+directory+names[0]+".csv};\n")
    text_lines.append("\\addlegendentry{\\texttt{val\\char`_acc}}\n")
    text_lines.append("\\end{axis}\n")
    text_lines.append("\\end{tikzpicture}\n")
    for i in range(1,len(names)):
        text_lines.append("\\newpage\n")
        text_lines.append("\\centerline{\\textbf{\\Huge "+ names[i]+"}}~\\vspace{0.4cm}\\\\")
        text_lines.append("\\begin{tikzpicture}\n")
        text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
        text_lines.append("\\addplot[color=blue,mark=*] table [x=epoch, y=loss, col sep=comma]{"+directory+names[i]+".csv};\n")
        text_lines.append("\\addlegendentry{loss}\n")
        text_lines.append("\\end{axis}\n")
        text_lines.append("\\end{tikzpicture}\\quad\\begin{tikzpicture}\n")
        text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
        text_lines.append("\\addplot[color=red,mark=*] table [x=epoch, y=val_loss, col sep=comma]{"+directory+names[i]+".csv};\n")
        text_lines.append("\\addlegendentry{\\texttt{val\\char`_loss}}\n")
        text_lines.append("\\end{axis}\n")
        text_lines.append("\\end{tikzpicture}\\\\")
        text_lines.append("\\begin{tikzpicture}\n")
        text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
        text_lines.append("\\addplot[color=teal,mark=*] table [x=epoch, y=acc, col sep=comma]{"+directory+names[i]+".csv};\n")
        text_lines.append("\\addlegendentry{acc}\n")
        text_lines.append("\\end{axis}\n")
        text_lines.append("\\end{tikzpicture}\\quad\\begin{tikzpicture}\n")
        text_lines.append("\\begin{axis}[legend style={at={(0.5,-0.1)},anchor=north}]\n")
        text_lines.append("\\addplot[color=magenta,mark=*] table [x=epoch, y=val_acc, col sep=comma]{"+directory+names[i]+".csv};\n")
        text_lines.append("\\addlegendentry{\\texttt{val\\char`_acc}}\n")
        text_lines.append("\\end{axis}\n")
        text_lines.append("\\end{tikzpicture}\n")
    text_lines.append("\\end{document}")
    file=open("test_latex.txt","w")
    file.writelines(text_lines)
    file.close()
    




def main():
    periods,sizes,maxPolyDegrees,polyWeightVariances,epochs,learning_rate,batch_size,experiments_per_dataset=get_input()
    for i in range(len(periods)):
            CNN(periods[i],sizes[i],maxPolyDegrees[i],polyWeightVariances[i],epochs[i],learning_rate[i],batch_size[i],experiments_per_dataset[i])
    answer=input("Create directory folder for storing csv files? y/n")
    while(True):
        if answer=="y" or answer=="n":
            break
        else:
            answer=input("Create directory folder? y/n")
    if answer=="y":
        dir_path=input("directory folder's name")
        os.mkdir(dir_path)
    else:
        dir_path=""
    filepath=input("enter the path for txt file")
    create_csv(fliepath,dir_path)
    create_latex_txt()


main()
