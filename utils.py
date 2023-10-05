import os
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy
import matplotlib.pyplot as plt

def write_accuracy_to_xls(data,timepoint,path,prefix):
    writepath = os.path.join(path, 'accurate'+prefix+'.xls')
    if not os.path.exists(writepath):
        wk = xlwt.Workbook()
        wksheet = wk.add_sheet('sheet1')
        # write index label
        for i in range(np.array(data[0]).shape[0]):
            wksheet.write(i+1, 0, 'T'+str(i+1))
        wk.save(writepath)

    # write in append mode
    oldWb = xlrd.open_workbook(writepath)
    newWb = copy(oldWb)
    newWs = newWb.get_sheet(0)
    # write column label
    newWs.write(0,timepoint,'Step'+str(timepoint))
    # write new line
    for i in range(np.array(data[0]).shape[0]):
        newWs.write(i + 1, timepoint, data[0][i])
    newWb.save(writepath)

def write_accuracies_to_xls(path, timepoint, accuracy_M1, accuracy_M2, accuracy_M3, accuracy_Ensem, conf1, conf2):
   # write averaged accuracy to accuracy.xls file
    if not os.path.exists(path):
        os.makedirs(path)
    
    # write mean accuracy to accuracy.xls files
    writepath = os.path.join(path, "accurate.xls")
    if not os.path.exists(writepath):
        wk = xlwt.Workbook()
        sh = wk.add_sheet("shee1")
        # write headline
        sh.write(0,0,'meanAcc_M1')
        sh.write(0,1,'meanAcc_M2')
        sh.write(0,2,'meanAcc_M3')
        sh.write(0,3,'meanAcc_Ensem')
        sh.write(0,6,'conf1')
        sh.write(0,7,'conf2')
        wk.save(writepath)

    # write new line
    oldWb = xlrd.open_workbook(writepath)
    newWb = copy(oldWb)
    newWs = newWb.get_sheet(0)
    newWs.write(timepoint, 0, accuracy_M1[1])
    newWs.write(timepoint, 1, accuracy_M2[1])
    newWs.write(timepoint, 2, accuracy_M3[1])
    newWs.write(timepoint, 3, accuracy_Ensem[1])
    newWs.write(timepoint, 6, conf1)
    newWs.write(timepoint, 7, conf2)
    newWb.save(writepath)

    # write detailed accuracy to accuracy_M.xls files
    write_accuracy_to_xls(accuracy_M1, timepoint, path, '_M1')
    write_accuracy_to_xls(accuracy_M2, timepoint, path,'_M2')
    write_accuracy_to_xls(accuracy_M3, timepoint, path, '_M3')
    write_accuracy_to_xls(accuracy_Ensem, timepoint, path, '_Ensem')

def save_diffimage(writepath, recimage, prefix):
    plt.clf()
    outsize = np.sqrt(recimage.size).astype(np.int16)
    plt.imshow(recimage.reshape(outsize, -1), cmap="bwr")
    plt.colorbar()
    plt.savefig(os.path.join(writepath, 'avr_diff' + prefix + '.jpg'))
                
def save_diffimages(writepath,timepoint,recimage,recimage_M1,recimage_M2,recimage_M3):
    if not os.path.exists(writepath):
        os.makedirs(writepath)

    save_diffimage(writepath, recimage_M1, f'_Step{timepoint:04d}_M1')
    save_diffimage(writepath, recimage_M2, f'_Step{timepoint:04d}_M2')
    save_diffimage(writepath, recimage_M3, f'_Step{timepoint:04d}_M3')
    save_diffimage(writepath, recimage, f'_Step{timepoint:04d}_Ensem')

def save_average_diffimages(writepath,recimage,recimage_M1,recimage_M2,recimage_M3):
    if not os.path.exists(writepath):
        os.makedirs(writepath)

    print('recimage shape: ', len(recimage), 'x', recimage[0].shape)
    save_diffimage(writepath, np.array(recimage_M1).mean(axis=0), '_M1')
    save_diffimage(writepath, np.array(recimage_M2).mean(axis=0), '_M2')
    save_diffimage(writepath, np.array(recimage_M3).mean(axis=0), '_M3')
    save_diffimage(writepath, np.array(recimage).mean(axis=0), '_Ensem')
    