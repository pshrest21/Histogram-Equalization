import cv2
from collections import Counter
import matplotlib.pyplot as mplot
import numpy as np
from PIL import Image
  

#Read the images using openCV
fish = cv2.imread('fish.pgm', 0)
jet = cv2.imread('jet.pgm', 0)

def oneDList(img_array):    
    #create convert 2D array of images to 1D array
    new_img_array = [item for sublist in img_array for item in sublist] 
    return new_img_array

def img_dict(img_array):  
    new_img_array = oneDList(img_array)
    #make a dictionary where the key is the grayscale and value is it's count
    my_dict = dict(Counter(new_img_array))
   
    #sort the dictionary in ascending order and convert it to list of tuples of key and value pairs
    my_list = sorted(my_dict.items())

    #set the keys to x and values to y
    x,y = zip(*my_list)   
    x = np.array(x)
    y = np.array(y)
  
    my_new_dict = dict()
    #have key and value pairs for all grayscales (0-255)
    for i in range(0, 256):
        if(i in x):
            my_new_dict[i] = my_dict[i]
        else:
            my_new_dict[i] = 0      
     
    return my_new_dict

#returns a numpy array of cumulative density function's y-axis's values
def cdf(my_list, y):
    cdf_list = []
    sum = y[0]
    cdf_list.append(sum)
    for i in range(1, len(y)):
        sum = sum + y[i]
        cdf_list.append(sum)
    
    cdf_list = np.array(cdf_list)
    return cdf_list
 
    
#function that takes 2D list of image, upper bound and lower bound, and returns
#a 2D list whose each element is linearly stretched       
def contrastEnhance(img_array, l2, l1):    
    lower = l1 * 255
    upper = l2 * 255
    
    slope = 255 / (upper - lower)
    b = -1 * slope * lower
    
    for i in range (0, len(img_array)-1):
        for j in range(0, len(img_array[i])-1):
           
            if(img_array[i][j] < lower):
                img_array[i][j] = 0
            elif(img_array[i][j] > upper):
                img_array[i][j] = 255
            else:
                img_array[i][j] = slope * img_array[i][j] + b
                
    img_array = np.clip(img_array,0,255)  
    return img_array

#level slicing
def levelSlice(img_array, l):
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array[i])):
            if img_array[i][j] >= l and img_array[i][j] <= l+10:
                img_array[i][j] = 255
            else:
                img_array[i][j] = 0
    return img_array


#Histogram Equalization
def histEqualize(img_array, cdf_list):
    final_dict = dict()
    cdf_list = np.rint(cdf_list)
    #print(cdf_list)
    for i in range(0, len(cdf_list)):
        final_dict[i] = int(cdf_list[i])
    #print(final_dict)    
    my_list = np.zeros([512, 512])
    #print(img_array)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array[i])):
            my_list[i][j] = final_dict[img_array[i][j]]
            
    #print(my_list)
    return my_list
 

def displayPmf(x, y, xLabel, yLabel, title):
    mplot.bar(x, y)
    mplot.xlabel(xLabel)
    mplot.ylabel(yLabel)
    mplot.title(title)
    
    mplot.show()
 

def displayCdf(x, y, xLabel, yLabel, title):
    mplot.plot(x, y)
    mplot.xlabel(xLabel)
    mplot.ylabel(yLabel)
    mplot.title(title)  
    
    mplot.show()
  
def getImageInfo(img_array):
    #getting information from original image     
    dict_img = img_dict(img_array)
    my_list = sorted(dict_img.items())
    
    #set the keys to x and values to y
    x,y = zip(*my_list)   
    x = np.array(x)
    y = np.array(y) / 262144
    cdf_fish = cdf(x, y)
    
    return x,y,cdf_fish
    


#----Display original image, it's pmf and cdf---
#for fish.pgm       
fish_image = Image.fromarray(fish)
fish_image.convert('L').save('fish.png', optimize = True) 

x, y, main_cdf_fish = getImageInfo(fish)

#main_cdf_fish = main_cdf_fish / 262144
displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of 'fish.png'")
displayCdf(x, main_cdf_fish, "Gray Level (0-255)", "Cumulative Distribution of Gray Levels", "Cdf of 'fish.png'")

#for jet.pgm
jet_image = Image.fromarray(jet)
jet_image.convert('L').save('jet.png', optimize = True)

x,y,main_cdf_jet = getImageInfo(jet)
#main_cdf_jet = main_cdf_jet / 262144
displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of 'jet.png'")
displayCdf(x, main_cdf_jet, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Cdf of 'jet.png'")




#-------------Contrast Stretching--------------
#for fish
fish_array = contrastEnhance(fish, 0.9, 0.1)

img_fish = Image.fromarray(fish_array)
img_fish.convert('L').save('fish_stretched.png', optimize = True)
#problem here too
x,y,cdf_fish = getImageInfo(fish_array)


displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of Contrast Stretched Fish")


#for jet
jet_array = contrastEnhance(jet, 0.9, 0.1)
img_jet = Image.fromarray(jet_array)
img_jet.convert('L').save('jet_stretched.png', optimize = True)

x,y,cdf_jet = getImageInfo(jet_array)
displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of Contrast Stretched Jet")




#-------------Level Slicing--------------------
fish = cv2.imread('fish.pgm', 0)
jet = cv2.imread('jet.pgm', 0)
#for fish
img_array_fish = levelSlice(fish, 200)
img_fish = Image.fromarray(img_array_fish)
img_fish.convert('L').save('fish_sliced.png', optimize = True)

x,y,cdf_fish = getImageInfo(img_array_fish)
displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of Sliced Image Fish")


#for jet
img_array_jet = levelSlice(jet, 200)
img_jet = Image.fromarray(img_array_jet)
img_jet.convert('L').save('jet_sliced.png', optimize = True)

x,y,cdf_jet= getImageInfo(img_array_jet)
displayPmf(x, y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Pmf of Sliced Image Jet")






#-------------Histogram Equalization For Fish-----------   

#read the image again because when doing slicing, the original image array gets sliced into either 255 or 0 graylevel
fish = cv2.imread('fish.pgm', 0)
jet = cv2.imread('jet.pgm', 0)


new_fish = histEqualize(fish, main_cdf_fish * 255)
new_x, new_y, new_cdf_fish = getImageInfo(new_fish)


new_img_fish = Image.fromarray(new_fish)
new_img_fish.convert('L').save('fish_equalized.png', optimize = True)

displayPmf(new_x, new_y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Histogram of Equalized Image 'fish.png'")
displayCdf(new_x, new_cdf_fish, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Cdf of Equalized Image 'fish.png'")

#-------------Histogram Equalization For Jet-----------   

new_jet = histEqualize(jet, main_cdf_jet * 255)
new_x, new_y, new_cdf_jet = getImageInfo(new_jet)


new_img_jet = Image.fromarray(new_jet)
new_img_jet.convert('L').save('jet_equalized.png', optimize = True)

displayPmf(new_x, new_y, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Histogram of Equalized Image 'jet.png'")
displayCdf(new_x, new_cdf_jet, "Gray Level (0-255)", "Total Number of Pixels (0-1)", "Cdf of Equalized Image 'jet.png'")























