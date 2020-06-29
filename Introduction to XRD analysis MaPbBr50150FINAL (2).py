#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize 
import pandas as pd


# In[69]:


def csv_to_np(filename): #changing excel to numpy
    data = pd.read_csv(filename)
    return(np.array(data))

perov = csv_to_np(r'C:\Users\tiann\OneDrive\Desktop\Summer Research 2020\xrdactivitypeaks.csv')#include r to for Python to read '/'


# ### Exercise 1
# To test your import and plotting skills go ahead and write a piece of code in the cell below that will plot the first and last frames of XRD data from the csv we imported in the same figure.
# 
# 

# In[70]:


plt.figure(figsize=(8,6)) #make plot larger
plt.plot(perov[:,0],perov[:,1],'r-', label='$MAPbIBr50150$') #plot two-theta versus XRD intesntiy
#FEEDBACK This currently plots the first frame of data, if you want to add the last frame you can add another line of code and use the index [-1] to call the last element of a list or array i.e. perov[:,-1]
plt.xlabel('2-theta [$^o$]',size=12) 
plt.ylabel('Intensity [a.u.]',size=12)
plt.legend(loc="upper right")


# In[ ]:





# 

# In[71]:


#Note that our initial 2-theta values are in degrees, not radians
#We also need to define the X-ray wavelength we're using
wave = 0.9763 #wavelength used at SSRL in Angstroms
theta = perov[:,0]/2*np.pi/180 #convert from 2-theta to theta in radians
q = 4*np.pi*np.sin(theta)/wave #convert to q


# In[72]:


#We can now check that our conversion went as expected by ploting our data versus q
plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q,perov[:,1],'b-', label='$MAPbIBr50150$') #plot Q versus XRD intesntiy
plt.xlabel('Q [$\AA^{-1}$]',size=12)
plt.ylabel('Intensity [a.u.]',size=12)
plt.legend(loc="upper right")


# ## exercise 2
# This is a process we'd like to do every time we open a piece of XRD data. Given that, it's worthwhile to make a function that will do this conversion for us. Go ahead and write a function that takes in an array of 2-theta values and a X-ray wavelength $\lambda$, and returns an array of Q values.
# 
# 

# In[119]:


import math
wave = 0.9763
theta = perov[:,0]/2*np.pi/180
def two_to_q(two_theta, wave):
    two_theta, wave = np.asarray(two_theta, wave) #to put it as an array if it is not already
    radians = perov[:,0]/2*np.pi/180 #FEEDBACK perhaps you want to assign a different variable here to avoid confusion?
    Q= 4*np.pi*math.sin(radians)/wave
    return Q
print(q) #this seems correct?
#FEEDBACK Looks good to me! Though I often plot things to be sure


# In[116]:


plt.plot(Q, perov[:,1])


# In[74]:


def find_nearest(array, target):
    array = np.asarray(array) # In case the input is not an array type
    idx = (np.abs(array - target)).argmin() # Finds the index of the value closest to the target
    return idx
#We use our new function to find the index of real value closest to our desired limit
q = 4*np.pi*np.sin(theta)/wave
q_1 = 0.98 #lower limit for Q we'll consider
q_2 = 1.15 # upper limit - ideally there is only one peak between these values

limit1 = find_nearest(q, q_1) #First our lower limit
limit2 = find_nearest(q, q_2) #And of our higher limit


# In[75]:


print(limit1)
print(limit2)


# In[76]:


# Having extablished our limits we can now trim the data

q_sub = q[limit1:limit2] # We'll reduce the domain of Q
perov_sub = perov[limit1:limit2, (1,-1)] # And correspondingly shrink the size of our diffraction data set. 
#YOU NEED THE () ABOVE FOR (1,-1) OR ELSE THE CODE WILL NOT FUNCTION
#FEEDBACK Alternative syntax is 1:-1 - I often do this since its more similar to Matlab (my first love), but chose the thing that works for you!

# You'll also notice I'm dropping the first column of data in the perov array, since that's now accounted for in q_sub
#Now let's go ahead and plot again to see if our trimming worked
plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_sub[:,0],'g-', label='$MAPbIBr_2$') #plot subfield of data
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper right hand corner

q_linear = np.hstack((q_sub[0:10], q_sub[-11:-1])) #I'm taking the starting and ending values
perov_linear = np.hstack((perov_sub[0:10,0], perov_sub[-11:-1,0])) #We'll use these to fit a straight line
slope, intercept = np.polyfit(q_linear, perov_linear, 1) #Do linear fit
back = slope*q_sub+intercept #Create background array of the form Background = Ax+B
#print(back)
plt.plot(q_sub, back)


# ### Exercise 3

# In[77]:


def find_nearest(array, target):
    array = np.asarray(array) # In case the input is not an array type
    idx = (np.abs(array - target)).argmin() # Finds the index of the value closest to the target
    return idx
#We use our new function to find the index of real value closest to our desired limit
q = 4*np.pi*np.sin(theta)/wave
q_1 = 0.98 #lower limit for Q we'll consider
q_2 = 1.15 # upper limit - ideally there is only one peak between these values

limit1 = find_nearest(q, q_1) #First our lower limit
limit2 = find_nearest(q, q_2) #And of our higher limit


# In[78]:


# Using the linear fitting introduced in Tutorial 03, let's fit a straight line for the data around our peak.
# For total transparency this method could be MUCH MUCH better. Open to suggestions!

#mypoly= np.poly1d(slope_intercept)
#print(mypoly)
#xfit = np.linspace(.975, 1.150, num=100)
#yfit = mypoly(xfit)

plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_sub[:,0]-back,'g-', label='$MAPbIBr50150$') #plot subfield of data
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper right hand 
#plt.plot(xfit, yfit, 'r-', label = 'Best Fit')


# ##### I don't know how to remove this line

# In[79]:


# FEEDBACK You can delete cells in a jupyter notebook by clicking delete cell under the edit tab


# In[80]:


#Let's begin by getting our data ready to analyze
perov_fit = perov_sub[:,0]-back #We'll begin by subtracting the background we calculated for this piece of data
#perov sub is intensity
#Now let's define a function we'll want to fit to - this is analagous to the "straight-line-model" from tutorial 03
#We'll call our function gaussian and it will calculate the expression described above
def gaussian(x, a, b, c): 
    return a*np.exp(-(x - b)**2/(2*c**2))

#We'll also give an initial guess for our fits based off of a visual interpretaion of our data
p0 = [250, 1, 0.2]

#Use scipy.optimize.curve_fit to fit our desired data
popt, pcov = scipy.optimize.curve_fit(gaussian, q_sub, perov_fit, p0)


# In[81]:


#To confirm our fits it's always nice to plot our model versus our data.
plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_fit,'r-', label='$MAPbIBr_2$') #plot subfield of data
plt.plot(q_sub,gaussian(q_sub, *popt),'b--', label='Model') #plot best fit
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper left hand corner


# In[82]:


#With confidence in our fit we can now go ahead and print/make note of/tabulate our parameters of interest. 
#Print peak intensity
print('Intensity:', popt[0])

#Caculate and pring d-spacing
d = 2*np.pi/popt[1] #Applying d = 2*pi/Q
print('d-Spacing: ', d) 

#Print lattice constant
miller = [1, 0, 0] #need to guess miller indices of peak
a = d/np.sqrt(miller[0]**2+miller[1]**2+miller[2]**2) #calculate a using a = d/sqrt(h^2+k^2+l^2) for a cubic lattice
print('Lattice Spacing:', a)


# ### Exercise 4

# In[83]:


import numpy as np #what I used for last Friday's activity
import math 
h=3
k=1
l=1
a = 6.04
d = a/(((h**2+k**2+l**2))**.5)
print('d=', d)
lambbda = .9763
Q = (2*np.pi)/d
print('Q=', Q)
theta1 = math.asin((Q*lambbda)/(4*np.pi))
print('theta1', theta1)
answer = theta1*2
print('radian=', answer)
print('angle=', answer*180/np.pi)


# In[84]:


# Exercise 4
import math
wave = 0.9763
theta = perov[:,0]/2*np.pi/180
def two_to_q(two_theta, wave):
    two_theta, wave = np.asarray(two_theta, wave) #to put it as an array if it is not already
    two_theta = perov[:,0]/2*np.pi/180
    4*np.pi*math.sin(two_theta)/wave
    return q

lambbda = .9763

planes = [[1,0,0],[1,1,0], [1,1,1], [2,0,0], [3,0,0]]
#two_d = np.array([planes])
#print(two_d)
#print(two_d[:,1]**2)
for plane in planes: #hones in on one miller index
    a = 6.03 #FEEDBACK Better would be to link this directly to previous fit for maximum accuracy
    sumplane = 0
    for n in plane: #hones in on each value within the miller index #FEEDBACK NESTED FOR LOOPS! NICE!!
        millersqr = n**2
        sumplane += millersqr
    d = a/((sumplane)**.5)
    print('d=', d)
    print('miller index sum squared=', sumplane) #FEEDBACK Might want to print specific miller index for readability? personally I find it easier to interpret [110] than 2.
    lambbda = .9763
    Q = (2*np.pi)/d
    print('Q=', Q)
    theta1 = (math.asin((Q*lambbda)/(4*np.pi)))
    print('theta1', theta1)
    answer = theta1*2
    print('radian=', answer)
    print('angle=', answer*180/np.pi)
    print()

#FEEDBACK This is really good! Slight improvement would be to generate an array of q values to that they're easier to store and reference later for plotting/comparisson/etc. How could you build that array as you step through your outer loop? How could you then plot those values on top of the actual XRD data/compare them tp the XRD data. 


# In[90]:


def find_nearest(array, target):
    array = np.asarray(array) # In case the input is not an array type
    idx = (np.abs(array - target)).argmin() # Finds the index of the value closest to the target
    return idx
#We use our new function to find the index of real value closest to our desired limit
q = 4*np.pi*np.sin(theta)/wave
q_1 = 2.06 #lower limit for Q we'll consider
q_2 = 2.11 # upper limit - ideally there is only one peak between these values

limit1 = find_nearest(q, q_1) #First our lower limit
limit2 = find_nearest(q, q_2) #And of our higher limit


# In[91]:


# Having extablished our limits we can now trim the data

q_sub = q[limit1:limit2] # We'll reduce the domain of Q
perov_sub = perov[limit1:limit2, (1,-1)] # And correspondingly shrink the size of our diffraction data set. 
#YOU NEED THE () ABOVE FOR (1,-1) OR ELSE THE CODE WILL NOT FUNCTION

# You'll also notice I'm dropping the first column of data in the perov array, since that's now accounted for in q_sub
#Now let's go ahead and plot again to see if our trimming worked
plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_sub[:,0],'g-', label='$MAPbIBr_2$') #plot subfield of data
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper right hand corner

q_linear = np.hstack((q_sub[0:10], q_sub[-11:-1])) #I'm taking the starting and ending values
perov_linear = np.hstack((perov_sub[0:10,0], perov_sub[-11:-1,0])) #We'll use these to fit a straight line
slope, intercept = np.polyfit(q_linear, perov_linear, 1) #Do linear fit
back = slope*q_sub+intercept #Create background array of the form Background = Ax+B
plt.plot(q_sub, back)


# In[109]:


#Let's begin by getting our data ready to analyze
perov_fit = perov_sub[:,0]-back #We'll begin by subtracting the background we calculated for this piece of data

plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_sub[:,0]-back,'g-', label='$MAPbIBr_2$') #plot subfield of data
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper right hand corner


#Now let's define a function we'll want to fit to - this is analagous to the "straight-line-model" from tutorial 03
#We'll call our function gaussian and it will calculate the expression described above
def gaussian(x, a, b, c): 
    return a*np.exp(-(x - b)**2/(2*c**2))

#We'll also give an initial guess for our fits based off of a visual interpretaion of our data
p0 = [35, 2.06, 0.01]

#Use scipy.optimize.curve_fit to fit our desired data
popt, pcov = scipy.optimize.curve_fit(gaussian, q_sub, perov_fit, p0)


# ### Confused on what p0 stands for and how to manipulate these numbers to make a model curve
# 
# FEEDBACK p0 is the starting point for our guess. The optimize function is going to try and get the best fit for valyes a, b, and c and by default it starts with values of 1 for all of them. If that's far away from the actual values, the optimization can fail (i.e. the program takes a limited number of steps to try and get a good fit, and if it can't find a good fit within those time steps it gives up). p0 therefore improves the likelihood of success by overwriting those defaults, it says start with an a value of 35, a b value or 2.06 and a c value of .01 and then lets the program go from there. 

# In[110]:


#To confirm our fits it's always nice to plot our model versus our data.
plt.figure(figsize=(8,6)) #make plot larger
plt.plot(q_sub,perov_fit,'r-', label='$MAPbIBr_2$') #plot subfield of data
plt.plot(q_sub,gaussian(q_sub, *popt),'b--', label='Model') #plot best fit
plt.xlabel('Q [$\AA^{-1}$]',size=12) #Define x-axis label
plt.ylabel('Intensity [a.u.]',size=12)#Define y-axis label
plt.legend(loc="upper right")#Put legend in upper left hand corner

#FEEDBACK it looks like you're having an issue wiht the badkground substratcion for this peak (that's why it's sloping to the right). My guess is that it's coming from part of the peak falling into the bounds of what we're calling background. You chould try and adjust this my changing the range of values we're using for background subtraction i.e. q_sub[-5:-1] instead of q_sub[-11:-1]


# In[111]:


#With confidence in our fit we can now go ahead and print/make note of/tabulate our parameters of interest. 
#Print peak intensity
print('Intensity:', popt[0])

#Caculate and pring d-spacing
d = 2*np.pi/popt[1] #Applying d = 2*pi/Q
print('d-Spacing: ', d) 

#Print lattice constant
miller = [1, 0, 0] #need to guess miller indices of peak
#FEEDBACK Need to update miller indice to reflect the peak you are looking at. In this case it's the [200] not the [100] peak
a = d/np.sqrt(miller[0]**2+miller[1]**2+miller[2]**2) #calculate a using a = d/sqrt(h^2+k^2+l^2) for a cubic lattice
print('Lattice Spacing:', a)


# In[ ]:





# In[ ]:




