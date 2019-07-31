import struct
from PIL import Image
import math
import numpy as np
import scipy.io as sio
import os, time
import matplotlib.pyplot as plt
from matplotlib import cm


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                               MY AEDAT OBJECT CLASS
#
#  Paer library developed by Dario not working --> ValueError: invalid literal for int() with base 16 when trying to 
#  read some of the recording files (problem with jAER that adds new line and incorrect time-stamps)
#                paer lib: https://github.com/darioml/python-aer
#
#     --> Created my own (simpler) set of functions to read and use Address-Event Representation (AER) data in python
#  
#  AER data --> .aedat extention (recorded using DVS and jAER)
#  Dario: class aedata and class aefile; why two classes ?
#  Faustine: I made only one class (aedatObj) with more attributes and the methods needed to preprocess the data. 
#            load_data inspired from loadaerdat.py (https://github.com/SensorsINI/processAEDAT/tree/master/jAER_utils)
#            Other source of inspiration: https://github.com/GergelyBoldogkoi/SpinVision/tree/master/SpinVision 
#            Still wondering if I should use Matlab files (.mat) or AER files (.aedat) as input for the neural network
#            PROBABLY .MAT: CONVERSION TO ARRAYS REQUIRED; DONE ONCE AT BEGINNING --> FILE SAVED --> AVOIDS TO DO
#  CONVERSION EVERY TIME THE SIMU IS RUN...  
#
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

"""
obj := class aedatObj
data := {'filename': self.filename, 'x': self.x, 'y': self.y, 'pol': self.pol, 'ts': self.ts, 'dim': self.dim} : associative array (dictionary) --> useful to save to mat
mat --> +/- like data
"""

class aedatObj(object):
    obj_list = [] # class attibute (instead of instance attribute) --> list of aedatObj 
    data_list = [] # list of data

    def __init__(self, filename=None, max_events=1e6):
        if filename == None:
            self.filename = 'new_obj_not_from_file' # could leave it to None
            self.x, self.y, self.pol, self.ts = np.array([]), np.array([]), np.array([]), np.array([])
            self.video_t = 0 
            self.header = ''
            self.dim = (0,0)
        else:
            self.filename = filename
            self.x, self.y, self.pol, self.ts, self.video_t, self.header = self.load_data()
            self.dim = (128,128)
        self.max_events = max_events # always < 60 000 in my recordings
        # obj/data in list not modified when real obj modified --> pointeur ?
        # only add obj created from file? 
        if len(self.ts) > 0: # don't add empty obj (ex temporary obj in fct)
            aedatObj.obj_list.append(self)
            aedatObj.data_list.append({'filename': self.filename, 'x': self.x, 'y': self.y, 'pol': self.pol, 'ts': self.ts, 'dim': self.dim})
        
        
    def load_data(self, debug=1):
        """    
        load AER data file and parse these properties of AE events:
        - timestamps (in us), 
        - x,y-position [0..127]
        - polarity (0/1)
        @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
        @return (xaddr, yaddr, pol, timestamps, video_duration, header) 
                --> 4 lists containing data of all events, 1 float of video duration, 1 string for the header
        """
      # constants
        aeLen = 8  # 1 AE event takes 8 bytes
        readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
        td = 0.000001  # timestep is 1us   
    
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
        ###################################
        #y x p?

        datafile = self.filename
    
        aerdatafh = open(datafile, 'rb')
        k = 0  # line number
        p = 0  # pointer, position on bytes
        statinfo = os.stat(datafile)
        length = statinfo.st_size    
        print ("file size", length)
        #######################################
        #print file size?
    
        # header
        lt = aerdatafh.readline()
        header = ''
        while lt and lt[0] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline() 
            header = header + str(lt) + '\n'
            if debug >= 2:
                print (str(lt))
            continue
        ###############################
        #wait to see debug change
        #multiple line header
        
        header = str(lt)

        # variables to parse
        timestamps = np.array([])
        xaddr = np.array([])
        yaddr = np.array([])
        pol = np.array([])
        ######################
        #what is pol?-on-1 off-0 polar
    
        # read data-part of file
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
    
        print (xmask, xshift, ymask, yshift, pmask, pshift)  

        while p < length:
            addr, ts = struct.unpack(readMode, s)# s char 1 byte
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            if debug >= 2: 
                print("ts->", ts) 
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps = np.append(timestamps, ts)
            xaddr = np.append(xaddr, x_addr)
            yaddr = np.append(yaddr, y_addr)
            pol = np.append(pol, a_pol)
            #np array catch
                  
            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen        

        video_duration = (timestamps[-1] - timestamps[0]) * td#time duration

        if debug > 0:
            try:
                print 'file name =', self.filename
                print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
                n = 5
                print ("showing first %i:" % (n))
                print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
            except:
                print ("failed to print statistics")

        return xaddr, yaddr, pol, timestamps, video_duration, header


    def save_to_mat(self, mat_filename=None): # mat_filename=self.filename[0:3] not possible --> set default arg to None and define inside
        """    
        save the aedatObj as a .mat file
        --> all attributes are saved
        @param mat_filename - if None, save new .mat file with the same name as .aedat file
        """
        if mat_filename == None:
            mat_filename = self.filename[0:3]
        sio.savemat(mat_filename, {'filename': mat_filename, 'x': self.x, 'y': self.y, 'pol': self.pol, 'ts': self.ts, \
        'video_t': self.video_t, 'header': self.header, 'dim': self.dim, 'max_events': self.max_events}) # save all attributes?
        # filename[0:3] --> name = 1-1.mat and not 1-1.aedat.mat
        # remember in python index starts at 0 and in [start:end], idx start is included but end is not
    

    # From G - but does not produce anything (YET)
    #############################
    #want to show what????????
    def interactive_animation(self, step=5000, limits=(0,128), pause=0):
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        plt.show()
        ax = fig.add_subplot(111)

        start = 0
        end = step-1
        while(start < len(self.x)):
            ax.clear()
            ax.scatter(self.x[start:end], self.y[start:end], s = 20, c = self.pol[start:end], marker = 'o') #, cmap = cm.jet )
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            start += step
            end += step
            plt.draw()
            time.sleep(pause)
            plt.show() #added by F


    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    #                           DATA PRE-PROCESSING
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    def downsample(self, new_dim=(32,32)):
        """    
        downsample aedatObj--> reduce resolution
        @param new_dim - new desired dimensions 
        @return rtn - new aedatObj with lower resolution
        """
        # TODO
        # Make this cleaner
        assert self.dim[0]%new_dim[0] is 0
        assert self.dim[1]%new_dim[1] is 0

        rtn = aedatObj() 

        rtn.filename = self.filename + str(new_dim[0]) #
        rtn.ts = self.ts
        rtn.pol = self.pol
        rtn.video_t = self.video_t
        rtn.x = np.floor(self.x / (self.dim[0] / new_dim[0]))
        rtn.y = np.floor(self.y / (self.dim[1] / new_dim[1]))
        rtn.dim = (new_dim[0], new_dim[1])
        rtn.header = 'Downsample of file ' + self.filename + ' to dim = ' + str(new_dim[0])

        return rtn


    def filter_noise(self, windowSize=3, threshold=12, exposureTime_ms=20): # from Gergely
        """    
        filter noise out aedatObj: keep only events concerning rolling ball
            --> square mask of size (windowSize, windowSize) moved across entire "frame" (128,128)
                at each step, counts number of ON and OFF events falling within the mask during a given time (exposureTime)
                if number of events reaches threshold, middle pixel is set to ON or OFF event
        @param windowSize - mask size (default=2)
        @param threshold - minimum number of events within mask and time required to keep the spike (default=12)
        @param exposureTime_ms - duration considered as same "time unit" (default=20)
        
        @return rtn - new aedatObj with noise filtered out
        """
        # this function converts all events into ON events
        # Gergely:  windowSiwze = 5
        #           threshold = 13 (empirically determined)
        #           exposureTime_ms = 20 (DVS exposure time)
        # Faustine: windowSiwze = 2 
        #           threshold = 12 (empirically determined)
        #           exposureTime_ms = 20 (DVS exposure time) 

        # TODO
        # Create different modes (pol_type) and compare in learning: 
        #   - all converted to ON 
        #   - only ON / (only OFF) 
        #   - keep ON and OFF --> this one reauires 2x more connections in first layer
        # Optimize
     
        new_ts = []
        new_x = []
        new_y = []
        new_pol = []
        exposureTime_us = exposureTime_ms * 1000

        for i in range(len(self.ts)):
            #find all events within plus-minus expTime
            # compare their x,y values, then if there are more than 4 in a 4x4 window, produce an output spike.
            eventCounter = 0
            ONcounter = 0
            OFFcounter = 0
            #go down
            j = i
            while(j >= 0 and abs(self.ts[j] - self.ts[i]) < exposureTime_us):
                #ignore polarity of data for now
                xdiff = abs(self.x[j] - self.x[i])
                ydiff = abs(self.y[j] - self.y[i])
                if xdiff <= windowSize and ydiff <= windowSize:
                    eventCounter += 1
                    if(self.pol[j]):
                        OFFcounter += 1
                    else:
                        ONcounter += 1
                j -= 1
                
            #go up
            k = i + 1
            while (k < len(self.ts) and abs(self.ts[k] - self.ts[i]) < exposureTime_us):
                # ignore polarity of data for now
                xdiff = abs(self.x[k] - self.x[i])
                ydiff = abs(self.y[k] - self.y[i])
                if xdiff <= windowSize and ydiff <= windowSize:
                    eventCounter += 1
                    if (self.pol[k]):
                        OFFcounter += 1
                    else:
                        ONcounter += 1
                k += 1

            if eventCounter >= threshold:
                new_ts.append(self.ts[i])
                new_x.append(self.x[i])
                new_y.append(self.y[i])
                if OFFcounter > ONcounter:
                    new_pol.append(1) # Add OFF event
                else:
                    new_pol.append(0) # Add ON event

        rtn = aedatObj()

        rtn.filename = self.filename + '_filtered'
        rtn.ts = np.array(new_ts)
        rtn.x = np.array(new_x)
        rtn.y = np.array(new_y)
        rtn.pol = np.array(new_pol)
        rtn.video_t = self.video_t
        rtn.dim = self.dim
        rtn.header = 'Filtered file ' + self.filename + ' \n Threshold = ' + str(threshold) + ' \n Window size = (' + \
                    str(windowSize) +', ' + str(windowSize) + ' \n Exposure time = ' + str(exposureTime_ms) + ' ms'

        return rtn


##################################################################################################

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                           FUNCTIONS TO MANIPULATE CLASS BUT NOT METHODS OF CLASS
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def merge_all_obj(newFile=None, timeBetweenSamples_us=1000, save=True): # return aedatObj
    
    new_obj = aedatObj()
    if newFile == None:
        newFile = 'all_files'
    new_obj.filename = newFile 
    new_obj.dim = aedatObj.obj_list[0].dim

    for obj in aedatObj.obj_list: 
        if (obj.dim == new_obj.dim): #check if possible; to check that all have the same dim
            new_obj = merge_2aedatObj(new_obj, obj, newFile='', save=False)
    
    new_obj.header = 'all individual file objects merged into one long object containing data from all files'
    
    if save == True:
        new_obj.save_to_mat(newFile)

    return new_obj # /!\ Not added to aedatObj_list


def merge_all_mat(files, newFile, iteration=1, timeBetweenSamples_ms=300, save=True): # iteration: how much time in a row do I present each sample
    new_filename = newFile 
    new_x = files[0]['x']
    new_y = files[0]['y']
    new_pol = files[0]['pol']
    new_ts = files[0]['ts'][0]
    new_header = 'all files'
    new_dim = files[0]['dim']
    new_max_events = files[0]['max_events']
    it = 1
    #endPos_list = [int(files[0]['filename'][0][2])]

    while it < iteration:
        if files[0]['dim'].all() == new_dim.all(): #only done for 32x32 files        
            new_x = np.append(new_x, files[0]['x']) 
            new_y = np.append(new_y, files[0]['y'])
            new_pol = np.append(new_pol, files[0]['pol'])
                               
            shifted_ts = files[0]['ts'] + (new_ts[-1] + timeBetweenSamples_ms) # new_ts[-1] should be 3000000
            new_ts = np.append(new_ts, shifted_ts)
            #endPos_list.append(int(files[0]['filename'][0][2]))
            #print len(new_x), len(new_y), len(new_pol), len(new_ts)
            it += 1
    it = 0

    for matFile in files[1:]:
        while it < iteration:
            if matFile['dim'].all() == new_dim.all(): #only done for 32x32 files        
                new_x = np.append(new_x, matFile['x']) 
                new_y = np.append(new_y, matFile['y'])
                new_pol = np.append(new_pol, matFile['pol'])
                               
                shifted_ts = matFile['ts'] + (new_ts[-1] + timeBetweenSamples_ms) # new_ts[-1] should be 3000000
                new_ts = np.append(new_ts, shifted_ts)
                #endPos_list.append(int(matFile['filename'][0][2]))
                #print len(new_x), len(new_y), len(new_pol), len(new_ts)
                it += 1
        it = 0

    if save == True:
        sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
        'header': new_header, 'dim': new_dim, 'max_events': new_max_events})#, 'endPos_list': endPos_list})


def merge_2mat(mat1, mat2, newFile=None, timeBetweenSamples_us=1000, save=True): # from 2 .mat file
    #works
    """
    mat1 and mat2 are matlab files loaded outside of function: ex. mat1 = sio.loadmat('1-1.mat')
    """
    if newFile == None:
        newFile = mat1['filename'] + '_' + mat2['filename']

    if mat1['dim'].all() == mat2['dim'].all():
        new_filename = newFile 
        new_x = np.append(mat1['x'], mat2['x']) 
        new_y = np.append(mat1['y'], mat2['y'])
        new_pol = np.append(mat1['pol'], mat2['pol'])
        
        #print len(new_pol)

        shifted_ts = mat2['ts'] + (mat1['ts'][0][-1] + timeBetweenSamples_us) # shifted_ts should be an array of same length as file2.ts
                                             # values should be those of file2.ts each incremented by the length of file1.ts + separated time
        new_ts = np.append(mat1['ts'], shifted_ts)

        new_header = np.append(mat1['filename'], mat2['filename'])
        new_dim = mat1['dim']
        new_max_events = max(mat1['max_events'], mat2['max_events']) 

    if save == True:
        sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
         'header': new_header, 'dim': new_dim, 'max_events': new_max_events})
   
    return sio.loadmat(new_filename+'.mat') #only works if save = True


def merge_2aedatObj(obj1, obj2, newFile=None, timeBetweenSamples_us=1000, save=True): # obj = aedatObj - return aedatObj (can save .mat)
    #works
    new_obj = aedatObj()
    if newFile == None:
        newFile = obj1.filename + '_' + obj2.filename

    if obj1.dim == obj2.dim:
        new_obj.filename = newFile 
        new_obj.x = np.append(obj1.x, obj2.x) 
        new_obj.y = np.append(obj1.y, obj2.y)
        new_obj.pol = np.append(obj1.pol, obj2.pol)
        
        if len(obj1.ts) == 0:
            shift = 0
        else:
            shift = obj1.ts[-1]

        shifted_ts = obj2.ts + (shift + timeBetweenSamples_us) # shifted_ts should be an array of same length as file2.ts
                                             # values should be those of file2.ts each incremented by the length of file1.ts + separated time
        new_obj.ts = np.append(obj1.ts, shifted_ts)

        new_obj.header = obj1.filename + ' + ' + obj2.filename
        new_obj.dim = obj1.dim

    if save == True:
        new_obj.save_to_mat(newFile)
   
    return new_obj


def make_matrix(x, y, pol, dim=(128,128)):
    image = np.zeros(dim)
    events = np.zeros(dim)

    for i in range(len(x)):
        image[int(y[i]-1), int(x[i]-1)] -= pol[i] - 0.5
        events[int(y[i]-1), int(x[i]-1)] += 1

    # http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    np.seterr(divide='ignore', invalid='ignore')

    result = 0.5 + (image / events)
    result[events == 0] = 0.5
    return result


def create_pngs(xaddr, yaddr, pol, name, step=3000, dim=(128,128)):

    idx = 0
    start = 0
    end = step-1

    while(start < len(xaddr)):
        image = make_matrix(xaddr[start:end],yaddr[start:end],pol[start:end], dim=dim)
        img_arr = (image*255).astype('uint8')
        im = Image.fromarray(img_arr)
        im.save('temp_fig/' + name + '_' + str(dim[0]) + ("_%05d" % idx) + ".png")
        idx += 1

        start += step
        end += step
 

def create_1png(xaddr, yaddr, pol, name, time=3000, dim=(128,128)):
    # no step her: fixed time
    start = time-200
    end = time+200

    image = make_matrix(xaddr[start:end],yaddr[start:end],pol[start:end], dim=dim)
    img_arr = (image*255).astype('uint8')
    im = Image.fromarray(img_arr)
    im.save('temp_fig/' + name + '_' + str(dim[0]) + ".png")


def divide_3(matFile, cut1, cut2, save=True): # because I recorded 3x the same traj in a row but actually I want them individually

    new_filename1 = matFile['filename'][0] + '-1'
    new_x1 = matFile['x'][0][0:cut1-100]
    new_y1 = matFile['y'][0][0:cut1-100]
    new_pol1 = matFile['pol'][0][0:cut1-100]
    new_ts1 = matFile['ts'][0][0:cut1-100]-matFile['ts'][0][0]
    new_dim1 = matFile['dim'][0]
    new_header1 = matFile['filename'][0] + ' 1st traj'
    new_max_events1 = matFile['max_events'][0]

    new_filename2 = matFile['filename'][0] + '-2'
    new_x2 = matFile['x'][0][cut1+100:cut2-100]
    new_y2 = matFile['y'][0][cut1+100:cut2-100]
    new_pol2 = matFile['pol'][0][cut1+100:cut2-100]
    new_ts2 = matFile['ts'][0][cut1+100:cut2-100]-matFile['ts'][0][cut1+100]
    new_dim2 = matFile['dim'][0]
    new_header2 = matFile['filename'][0] + ' 2nd traj'
    new_max_events2 = matFile['max_events'][0]

    new_filename3 = matFile['filename'][0] + '-3'
    new_x3 = matFile['x'][0][cut2+100:]
    new_y3 = matFile['y'][0][cut2+100:]
    new_pol3 = matFile['pol'][0][cut2+100:]
    new_ts3 = matFile['ts'][0][cut2+100:]-matFile['ts'][0][cut2+100]
    new_dim3 = matFile['dim'][0]
    new_header3 = matFile['filename'][0] + ' 3rd traj'
    new_max_events3 = matFile['max_events'][0]

    if save == True:
        sio.savemat(new_filename1, {'filename': new_filename1, 'x': new_x1, 'y': new_y1, 'pol': new_pol1, 'ts': new_ts1, \
         'header': new_header1, 'dim': new_dim1, 'max_events': new_max_events1})
        sio.savemat(new_filename2, {'filename': new_filename2, 'x': new_x2, 'y': new_y2, 'pol': new_pol2, 'ts': new_ts2, \
         'header': new_header2, 'dim': new_dim2, 'max_events': new_max_events2})
        sio.savemat(new_filename3, {'filename': new_filename3, 'x': new_x3, 'y': new_y3, 'pol': new_pol3, 'ts': new_ts3, \
         'header': new_header3, 'dim': new_dim3, 'max_events': new_max_events3})


def add_pos(matFile, endPos_list):
    new_filename = matFile['filename'][0] + '_withpos'
    new_header = matFile['header'][0] + ' with end positions list'

    sio.savemat(new_filename, {'filename': new_filename, 'x': matFile['x'][0], 'y': matFile['y'][0], 'pol': matFile['pol'][0], 'ts': matFile['ts'][0], \
         'header': new_header, 'dim': matFile['dim'][0], 'max_events': matFile['max_events'][0], 'endPos_list': endPos_list})


def same_len(files, length): # all traj same length <=> ball different speed...
    for matFile in files:
        print matFile['filename'][0]
        new_filename = matFile['filename'][0] + '_len' + str(length)
        new_x = matFile['x'][0]
        new_y = matFile['y'][0]
        new_pol = matFile['pol'][0]
        new_ts = []
        for i in range(len(matFile['ts'][0])):
            new_ts.append(int(matFile['ts'][0][i]*length/matFile['ts'][0][-1]))
        new_dim = matFile['dim'][0]
        new_header = matFile['filename'][0] + ' 3s long'
        new_max_events = matFile['max_events'][0]

        sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
         'header': new_header, 'dim': new_dim, 'max_events': new_max_events})


def make_len(matFile, length):
    print matFile['filename'][0] 
    new_filename = matFile['filename'][0][0:16] #+ '_len' + str(length/1000)
    new_x = matFile['x'][0]
    new_y = matFile['y'][0]
    new_pol = matFile['pol'][0]
    new_ts = []
    for i in range(len(matFile['ts'][0])):
        new_ts.append(int(int(matFile['ts'][0][i])*length/int(matFile['ts'][0][-1])))

    new_dim = matFile['dim'][0]
    new_header = matFile['filename'][0] + str(length/1000) + 's'
    new_max_events = matFile['max_events'][0]
    sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
     'header': new_header, 'dim': new_dim, 'max_events': new_max_events, 'endPos_list': matFile['endPos_list']})


def convert_to_ms(matFile): # comment endPos if not present
    print matFile['filename'][0]
    new_filename = matFile['filename'][0] + '_ms'
    new_header = matFile['header'][0] + ' spikes at the ms scale'
    new_ts = [int(matFile['ts'][0][0]/1000)]
    new_x = [matFile['x'][0][0]]
    new_y = [matFile['y'][0][0]]
    new_pol = [matFile['pol'][0][0]]

    for i in range(len(matFile['ts'][0])-1):
        if (matFile['x'][0][i+1] != matFile['x'][0][i]) or (matFile['y'][0][i+1] != matFile['y'][0][i]) or (int(matFile['ts'][0][i+1]/1000) != int(matFile['ts'][0][i]/1000)):
            new_ts.append(int(matFile['ts'][0][i+1]/1000))
            new_x.append(int(matFile['x'][0][i+1]))
            new_y.append(int(matFile['y'][0][i+1]))
            new_pol.append(int(matFile['pol'][0][i+1]))

    sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
         'header': new_header, 'dim': matFile['dim'][0], 'max_events': matFile['max_events'][0]}) #, 'endPos_list': matFile['endPos_list'][0]


def keep_only_ON_events(matFile):
    new_filename = matFile['filename'][0] + '_ON'
    new_header = matFile['header'][0] + ' Only ON events'
    new_ts = []
    new_x = []
    new_y = []
    new_pol = []

    for i in range(len(matFile['pol'][0])):
        if matFile['pol'][0][i] == 1:
            new_ts.append(matFile['ts'][0][i])
            new_x.append(matFile['x'][0][i])
            new_y.append(matFile['y'][0][i])
            new_pol.append(matFile['pol'][0][i])

    sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
         'header': new_header, 'dim': matFile['dim'][0], 'max_events': matFile['max_events'][0]}) #, 'endPos_list': matFile['endPos_list'][0]

def make_halftraj(matFile):
    new_filename = 'half_' + matFile['filename'][0] 
    new_x = matFile['x'][0][0:int(len(matFile['ts'][0])/2)]
    new_y = matFile['y'][0][0:int(len(matFile['ts'][0])/2)]
    new_pol = matFile['pol'][0][0:int(len(matFile['ts'][0])/2)]
    new_ts = matFile['ts'][0][0:int(len(matFile['ts'][0])/2)]

    new_dim = matFile['dim'][0]
    new_header = 'half trajectory of ' + matFile['filename'][0]
    new_max_events = matFile['max_events'][0]
    sio.savemat(new_filename, {'filename': new_filename, 'x': new_x, 'y': new_y, 'pol': new_pol, 'ts': new_ts, \
     'header': new_header, 'dim': new_dim, 'max_events': new_max_events, 'endPos_list': matFile['endPos_list']})


##########################


matFiles1 = []
matFiles2 = []
"""

for i in range(len(files)):
    file1 = [sio.loadmat(files[i])]
    merge_all_mat(file1, files[i][-18:-7], iteration=18)
    file1 = sio.loadmat(files[i][-18:-7])
    keep_only_ON_events(file1)
    file1 = sio.loadmat(files[i][-18:-7]+'_ON')
    convert_to_ms(file1)
for i in range(len(from24to3)):
    matFiles2.append(sio.loadmat(from24to3[i]))

merge_all_mat(matFiles1, 'from24to1-12')
matFile1 = sio.loadmat('from24to1-12')
keep_only_ON_events(matFile1)
matFile1 = sio.loadmat('from24to1-12_ON')
convert_to_ms(matFile1)

merge_all_mat(matFiles2, 'from24to3-12')
matFile2 = sio.loadmat('from24to3-12')
keep_only_ON_events(matFile2)
matFile2 = sio.loadmat('from24to3-12_ON')
convert_to_ms(matFile2)
"""
#sample = sio.loadmat('../Data_records/15mat_files/3indiv_traj/3rd_layer/from135_random_3s_withpos_5pres-12_ms_ON.mat')
"""
organisedData = {}
spikeTimes = []
    
for i in range(len(sample['ts'][0])):
    neuronId = (sample['x'][0][i], sample['y'][0][i])
    if neuronId not in organisedData:
        organisedData[neuronId] = [sample['ts'][0][i]/1000]#-sample['ts'][0][0]] # - not necessary in 3indiv_traj files
            # divided by 1000 because AEDAT was in us and SpiNNaker is in ms
            # I don't know how SpyNNkaer cope with 2 spikes at the same ms (different us but how does it know?) 
    else:
        organisedData[neuronId].append(sample['ts'][0][i]/1000)#-sample['ts'][0][0]) # - not necessary in 3indiv_traj files

for neuronSpikes in organisedData.values():
    neuronSpikes.sort()
    spikeTimes.append(neuronSpikes)

diff = []

for i in range(len(spikeTimes)):
    for j in range(len(spikeTimes[i])-1):
        if (spikeTimes[i][j+1] - spikeTimes[i][j]) > 1:
            diff.append(spikeTimes[i][j+1] - spikeTimes[i][j])

print max(diff), min(diff)"""

#make_len(sample, 100000)
"""
files = ['../Data_records/15AER_records/1-1.aedat']#, \
         '../Data_records/15AER_records/1-2.aedat', \
         '../Data_records/15AER_records/1-3.aedat', \
         '../Data_records/15AER_records/2-1.aedat', \
         '../Data_records/15AER_records/2-2.aedat', \
         '../Data_records/15AER_records/2-3.aedat', \
         '../Data_records/15AER_records/3-1.aedat', \
         '../Data_records/15AER_records/3-2.aedat', \
         '../Data_records/15AER_records/3-3.aedat', \
         '../Data_records/15AER_records/4-1.aedat', \
         '../Data_records/15AER_records/4-2.aedat', \
         '../Data_records/15AER_records/4-3.aedat', \
         '../Data_records/15AER_records/5-1.aedat', \
         '../Data_records/15AER_records/5-2.aedat', \
         '../Data_records/15AER_records/5-3.aedat']

"""



#merge_all_mat(matfiles, 'all_halftrajto13')

#file1 = sio.loadmat('../Data_records/15mat_files/3indiv_traj/alltraj1file/alltraj_3s_1it.mat')
#keep_only_ON_events(file1)

#file1 = sio.loadmat('alltraj_3s_1it_ON.mat')
#convert_to_ms(file1)
