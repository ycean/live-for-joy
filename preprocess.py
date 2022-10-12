#!/usr/bin/env python3
import csv
import numpy as np
import sys

################## split each joint into one csv file each ########################################
L1_j1 = np.array([])
L1_j2 = np.array([])
L1_j3 = np.array([])
L2_j1 = np.array([])
L2_j2 = np.array([])
L2_j3 = np.array([])
L3_j1 = np.array([])
L3_j2 = np.array([])
L3_j3 = np.array([])
L4_j1 = np.array([])
L4_j2 = np.array([])
L4_j3 = np.array([])
k = 1
with open("/home/ycean/Documents/backup/raw_data_0623/overall_label.csv",'r',newline ='') as file:
    reader = csv.reader(file,delimiter=',')
    for row in list(reader):
        L1_j1 = np.append(L1_j1,row[0])
        L1_j2 = np.append(L1_j2,row[1])
        L1_j3 = np.append(L1_j3,row[2])
        L2_j1 = np.append(L2_j1,row[3])
        L2_j2 = np.append(L2_j2,row[4])
        L2_j3 = np.append(L2_j3,row[5])
        L3_j1 = np.append(L3_j1,row[6])
        L3_j2 = np.append(L3_j2,row[7])
        L3_j3 = np.append(L3_j3,row[8])
        L4_j1 = np.append(L4_j1,row[9])
        L4_j2 = np.append(L4_j2,row[10])
        L4_j3 = np.append(L4_j3,row[11])
        print(k)
        k += 1


np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg1_j1.csv",L1_j1,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg1_j2.csv",L1_j2,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg1_j3.csv",L1_j3,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg2_j1.csv",L2_j1,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg2_j2.csv",L2_j2,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg2_j3.csv",L2_j3,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg3_j1.csv",L3_j1,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg3_j2.csv",L3_j2,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg3_j3.csv",L3_j3,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg4_j1.csv",L4_j1,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg4_j2.csv",L4_j2,fmt = "%s",delimiter=",")
np.savetxt("/home/ycean/Documents/backup/raw_data_0623/leg4_j3.csv",L4_j3,fmt = "%s",delimiter=",")
#################### for spearate each start and goal position into individual file _have to combine every test data into one file###############
################################# FOR Front data ###################################
#overall_input = np.array([])
#n = 0
#print("FRONT/front")
#k =1
#a = np.array([])
#c = np.array([])
#front_input = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/FRONT/front/input.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,4)
#        k = k+1
#
#front_p = k-1
#front_q = 0
############___________ front_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/FRONT/front_"+ str(i+1) + "/input.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,4)
#            k = k+1
#
#    front_q += k-1
#    c = (np.append(c,b)).reshape(front_q,4)
#
#front_n = front_p + front_q
#front_input = (np.append(a,c)).reshape(front_n,4)
#print(front_n)
#n += front_n
#overall_input = (np.append(overall_input,front_input)).reshape(n,4)
################################ FOR LEFT data ###################################
#print("LEFT/left")
#k =1
#a = np.array([])
#c = np.array([])
#left_input = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/LEFT/left/input.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,4)
#        k = k+1
#
#left_p = k-1
#left_q = 0
############___________ left_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/LEFT/left_"+ str(i+1) + "/input.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,4)
#            k = k+1
#
#    left_q += k-1
#    c = (np.append(c,b)).reshape(left_q,4)
#
#left_n = left_p + left_q
#left_input = (np.append(a,c)).reshape(left_n,4)
#print(left_n)
#n += left_n
#overall_input = (np.append(overall_input,left_input)).reshape(n,4)
################################## FOR right data ###################################
#print("RIGHT/right")
#k =1
#a = np.array([])
#c = np.array([])
#right_input = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/RIGHT/right/input.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,4)
#        k = k+1
#
#right_p = k-1
#right_q = 0
############___________ right_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/RIGHT/right_"+ str(i+1) + "/input.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,4)
#            k = k+1
#
#    right_q += k-1
#    c = (np.append(c,b)).reshape(right_q,4)
#
#right_n = right_p + right_q
#right_input = (np.append(a,c)).reshape(right_n,4)
#print(right_n)
#n += right_n
#overall_input = (np.append(overall_input,right_input)).reshape(n,4)
################################# FOR back data ###################################
#print("BACK/back")
#k =1
#a = np.array([])
#c = np.array([])
#back_input = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/BACK/back/input.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,4)
#        k = k+1
#
#back_p = k-1
#back_q = 0
############___________ back_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/input.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,4)
#            k = k+1
#
#    back_q += k-1
#    c = (np.append(c,b)).reshape(back_q,4)
#
#back_n = back_p + back_q
#back_input = (np.append(a,c)).reshape(back_n,4)
#print(back_n)
############## Overall data combine ######################
#n += back_n
#overall_input = (np.append(overall_input,back_input)).reshape(n,4)
#
#np.savetxt("/home/ycean/Documents/backup/raw_data_0623/overall_input.csv",overall_input,fmt = "%s,%s,%s,%s",delimiter=",")
#print(n)
##################### for spearate each joint data into individual file _have to combine every test data into one file###############
################################# FOR Front data ###################################
#overall_label = np.array([])
#n = 0
#print("FRONT/front")
#k =1
#a = np.array([])
#c = np.array([])
#front_label = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/FRONT/front/label.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,12)
#        k = k+1

#front_p = k-1
#front_q = 0
############___________ front_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/FRONT/front_"+ str(i+1) + "/label.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,12)
#            k = k+1
#
#    front_q += k-1
#    c = (np.append(c,b)).reshape(front_q,12)
#
#front_n = front_p + front_q
#front_label = (np.append(a,c)).reshape(front_n,12)
#print(front_n)
#n += front_n
#overall_label = (np.append(overall_label,front_label)).reshape(n,12)
################################ FOR LEFT data ###################################
#print("LEFT/left")
#k =1
#a = np.array([])
#c = np.array([])
#left_label = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/LEFT/left/label.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,12)
#        k = k+1
#
#left_p = k-1
#left_q = 0
############___________ left_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/LEFT/left_"+ str(i+1) + "/label.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,12)
#            k = k+1
#
#    left_q += k-1
#    c = (np.append(c,b)).reshape(left_q,12)
#
#left_n = left_p + left_q
#left_label = (np.append(a,c)).reshape(left_n,12)
#print(left_n)
#n += left_n
#overall_label = (np.append(overall_label,left_label)).reshape(n,12)
################################# FOR right data ###################################
#print("RIGHT/right")
#k =1
#a = np.array([])
#c = np.array([])
#right_label = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/RIGHT/right/label.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,12)
#        k = k+1
#
#right_p = k-1
#right_q = 0
############___________ right_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/RIGHT/right_"+ str(i+1) + "/label.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,12)
#            k = k+1
#
#    right_q += k-1
#    c = (np.append(c,b)).reshape(right_q,12)
#
#right_n = right_p + right_q
#right_label = (np.append(a,c)).reshape(right_n,12)
#print(right_n)
#n += right_n
#overall_label = (np.append(overall_label,right_label)).reshape(n,12)
################################# FOR back data ###################################
#print("BACK/back")
#k =1
#a = np.array([])
#c = np.array([])
#back_label = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/BACK/back/label.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,12)
#        k = k+1
#
#back_p = k-1
#back_q = 0
############___________ back_data ___________############
#for i in range (100):
#    k = 1
#    b = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/label.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            b = (np.append(b,row1)).reshape(k,12)
#            k = k+1
#
#    back_q += k-1
#    c = (np.append(c,b)).reshape(back_q,12)
#
#back_n = back_p + back_q
#back_label = (np.append(a,c)).reshape(back_n,12)
#print(back_n)

############## Overall data combine ######################
#n += back_n
#overall_label = (np.append(overall_label,back_label)).reshape(n,12)

#np.savetxt("/home/ycean/Documents/backup/raw_data_0623/overall_label.csv",overall_label,fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",delimiter=",")
#print(n)
####################  for first data of AC_COM _save in sequence.csv#######################
#print("BACK/back")
#k =1
#l =1
#a = np.array([])
#b = np.array([])
#se = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/BACK/back/AC_COM_POSE_X.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader):
#        a = (np.append(a,row)).reshape(k,1)
#        k = k+1
#with open("/home/ycean/Documents/backup/raw_data_0623/BACK/back/AC_COM_POSE_Y.csv",'r',newline ='') as file2:
#    reader2 = csv.reader(file2,delimiter=',')
#    for row2 in list(reader2):
#        b = (np.append(b,row2)).reshape(l,1)
#        l = l+1
#
#p = np.hstack((a,b))
#for j in range (int(len(p)/240)):
#    r = p[(j*240):240*(1+j),:]
#    se = (np.append(se,r)).reshape((j+1)*240,2)
#np.savetxt("/home/ycean/Documents/backup/raw_data_0623/BACK/back/sequence.csv",se,fmt = "%s,%s",delimiter=",",header ="COM_X,COM_Y")
#print(str(0)+':'+str(j+1))
#
#############___________ AC_COM data preprocessing_save in sequence.csv ___________############
#for i in range (100):
#    k = 1
#    l = 1
#    a = np.array([])
#    b = np.array([])
#    se = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/AC_COM_POSE_X.csv"
#    file_name2 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/AC_COM_POSE_Y.csv"
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            a = (np.append(a,row1)).reshape(k,1)
#            k = k+1
#    with open(file_name2,'r',newline ='') as file2:
#        reader2 = csv.reader(file2,delimiter=',')
#        for row2 in list(reader2):
#            b = (np.append(b,row2)).reshape(l,1)
#            l = l+1
#
#    q = np.hstack((a,b))
#
#    for m in range (int(len(q)/240)):
#        r = q[(m*240):240*(1+m),:]
#        se = (np.append(se,r)).reshape((m+1)*240,2)
#
#    save_name = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/sequence.csv"
#    np.savetxt(save_name,se,fmt = "%s,%s",delimiter=",",header ="COM_X,COM_Y")
#    print(str(i+1)+':'+str(m+1))
#############___________ input data preprocessing_save in input.csv ___________############
#for i in range (100):
#    k =1
#    a = np.array([])
#    file_name = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/input_label.csv"
#    with open(file_name,'r',newline ='') as file:
#        reader = csv.reader(file,delimiter=',')
#        for row in list(reader)[::60]:
#            a = (np.append(a,row)).reshape(k,4)
#            k = k+1
#
#    save_name = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/input.csv"
#    np.savetxt(save_name,a,fmt = "%s,%s,%s,%s",delimiter=",")
#
####################  for first data of input_save in input.csv #######################
#k =1
#a = np.array([])
#with open("/home/ycean/Documents/backup/raw_data_0623/BACK/back/input_label.csv",'r',newline ='') as file:
#    reader = csv.reader(file,delimiter=',')
#    for row in list(reader)[::60]:
#        a = (np.append(a,row)).reshape(k,4)
#        k = k+1
#
#np.savetxt("/home/ycean/Documents/backup/raw_data_0623/BACK/back/input.csv",a,fmt = "%s,%s,%s,%s",delimiter=",")
#
###############____________joint data preprocessing_save in both input.csv and label.csv________________############
#print("BACK/back")
#for i in range (100):
#    k = 1
#    l = 1
#    m = 1
#    n = 1
#    o = 1
#    a = np.array([])
#    b = np.array([])
#    c = np.array([])
#    d = np.array([])
#    e = np.array([])
#    label = np.array([])
#    input = np.array([])
#    file_name1 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/L1_J.csv"
#    file_name2 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/L2_J.csv"
#    file_name3 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/L3_J.csv"
#    file_name4 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/L4_J.csv"
#    file_name5 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/input.csv"
#
#    with open(file_name1,'r',newline ='') as file1:
#        reader1 = csv.reader(file1,delimiter=',')
#        for row1 in list(reader1):
#            a = (np.append(a,row1)).reshape(k,3)
#            k = k+1
#    with open(file_name2,'r',newline ='') as file2:
#        reader2 = csv.reader(file2,delimiter=',')
#        for row2 in list(reader2):
#            b = (np.append(b,row2)).reshape(l,3)
#            l = l+1
#    with open(file_name3,'r',newline ='') as file3:
#        reader3 = csv.reader(file3,delimiter=',')
#        for row3 in list(reader3):
#            c = (np.append(c,row3)).reshape(m,3)
#            m = m+1
#    with open(file_name4,'r',newline ='') as file4:
#        reader4 = csv.reader(file4,delimiter=',')
#        for row4 in list(reader4):
#            d = (np.append(d,row4)).reshape(n,3)
#            n = n+1
#    with open(file_name5,'r',newline ='') as file5:
#        reader5 = csv.reader(file5,delimiter=',')
#        for row5 in list(reader5):
#            e = (np.append(e,row5)).reshape(o,4)
#            o = o+1
#
#    q = np.hstack((a,b,c,d))
#
#    for j in range (int((len(q))/240)):
#        f = q[(j*240):240*(1+j),:]
#        label = (np.append(label,f)).reshape((j+1)*240,12)
#        g = e[j,:]
#        input = (np.append(input,g)).reshape((j+1),4)
#
#    #save_name_input = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/input.csv"
#    #save_name_label = "/home/ycean/Documents/backup/raw_data_0623/BACK/back_"+ str(i+1) + "/label.csv"
#
#    #np.savetxt(save_name_input,input,fmt = "%s,%s,%s,%s",delimiter=",")
#    #np.savetxt(save_name_label,label,fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",delimiter=",")
#    print(str(i)+':'+str(j+1))
#    ######_________first data of the label_save in both input.csv and label.csv _____________#############
#k = 1
#l = 1
#m = 1
#n = 1
#o = 1
#a = np.array([])
#b = np.array([])
#c = np.array([])
#d = np.array([])
#e = np.array([])
#input = np.array([])
#label = np.array([])
#file_name1 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/L1_J.csv"
#file_name2 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/L2_J.csv"
#file_name3 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/L3_J.csv"
#file_name4 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/L4_J.csv"
#file_name5 = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/input.csv"
#with open(file_name1,'r',newline ='') as file1:
#    reader1 = csv.reader(file1,delimiter=',')
#    for row1 in list(reader1):
#        a = (np.append(a,row1)).reshape(k,3)
#        k = k+1
#with open(file_name2,'r',newline ='') as file2:
#    reader2 = csv.reader(file2,delimiter=',')
#    for row2 in list(reader2):
#        b = (np.append(b,row2)).reshape(l,3)
#        l = l+1
#with open(file_name3,'r',newline ='') as file3:
#    reader3 = csv.reader(file3,delimiter=',')
#    for row3 in list(reader3):
#        c = (np.append(c,row3)).reshape(m,3)
#        m = m+1
#with open(file_name4,'r',newline ='') as file4:
#    reader4 = csv.reader(file4,delimiter=',')
#    for row4 in list(reader4):
#        d = (np.append(d,row4)).reshape(n,3)
#        n = n+1
#with open(file_name5,'r',newline ='') as file5:
#    reader5 = csv.reader(file5,delimiter=',')
#    for row5 in list(reader5):
#        e = (np.append(e,row5)).reshape(o,4)
#        o = o+1
#
#label_pre = np.hstack((a,b,c,d))
#
#for j in range (int(len(label_pre)/240)):
#    label_process = label_pre[(j*240):240*(1+j),:]
#    label = (np.append(label,label_process)).reshape((j+1)*240,12)
#    input_process = e[j,:]
#    input = (np.append(input,input_process)).reshape((j+1),4)

##save_name_input = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/input.csv"
##save_name_label = "/home/ycean/Documents/backup/raw_data_0623/BACK/back/label.csv"
##np.savetxt(save_name_input,input,fmt = "%s,%s,%s,%s",delimiter=",")
##np.savetxt(save_name_label,label,fmt = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",delimiter=",")
#print(j+1)
