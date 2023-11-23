# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:13:50 2023

@author: noga mudrik
"""
import numpy as np
import matplotlib.pyplot as plt

def checkEmptyList(obj):
    """
    Check if the given object is an empty list.

    Args:
        obj (object): Object to be checked.

    Returns:
        bool: True if the object is an empty list, False otherwise.

    """    
    return isinstance(obj, list) and len(obj) == 0


    
def create_colors(len_colors, perm = [0,1,2], style = 'random', cmap  = 'viridis', seed = 0, reduce_green = 0.4):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    np.random.seed(seed)
    if style == 'random':
        colors = np.random.rand(3, len_colors)
        colors[1] = colors[1]*reduce_green
    else:
        cmap = plt.get_cmap(cmap) 
        # Create an array of values ranging from 0 to 1 to represent positions in the colormap
        positions = np.linspace(0, 1, len_colors)

        colors = [cmap(pos) for pos in positions]

    return colors


def to_create_plot_of_cs(c, mat, colors = [], cmap  = 'viridis', epsilon = 1e-9, fig = [], ax = [], wind = 3, len_curve = 0.01,
                         plot_curves = False, closed_loop = True, seed = 0):

        
    if checkEmptyList(colors):
        colors = create_colors(c.shape[0], style = 'random', cmap = cmap, seed = seed)
        colors = [colors[:,i] for i in range(colors.shape[1])]
    c = np.abs(c)
   
    if len(mat.flatten()) == np.max(mat.shape):
        mat = mat.reshape((1,-1))
        
    if len(c.flatten()) == np.max(c.shape):
        c = c.reshape((1,-1))
    
        
        
    if mat.shape[1] != c.shape[1]:
        raise ValueError('size must match')
        
    if mat.shape[0] == 1 or len(mat.flatten()) == np.max(mat.shape):
        if checkEmptyList(ax):
            fig, ax = plt.subplots()        
        x = np.arange(len(mat.flatten()))
        y = mat.flatten()        
        y_end = y + c        
        ys = np.vstack([y, y_end])        
        [ax.fill([x_i, x_i, x[i+1], x[i+1]], [ys[0,i], ys[1,i], ys[1,i+1], ys[0,i+1]], color = colors[0], alpha = 0.2) for i, x_i in enumerate(x[:-1])]

        
    elif mat.shape[0] == 2:
        if checkEmptyList(ax):
            fig, ax = plt.subplots()
        
        x = mat[0].flatten()    
        y = mat[1].flatten()  
        
        for dim_c in range(c.shape[0]):          
            
    
            m_inv = np.vstack([-np.diff(y) , np.diff(x)])
            m_inv_l2 = np.sum(m_inv**2,0)**0.5
            m_inv = len_curve *m_inv/m_inv_l2.reshape((1,-1))
            m_inv = np.hstack([m_inv, m_inv[:,-1].reshape((-1,1)) ])
            
            endings = np.vstack([x,y])+ m_inv*np.repeat(c[dim_c].reshape((1,-1)), 2, axis = 0)


            starts = np.vstack([x,y])#[:,:-1]
            
            #ax.fill(np.hstack([starts[0], endings[0] ]), np.hstack([  starts[1] , endings[1] ]) , alpha = 0.5, color = colors[dim_c]) 
            # ax.scatter(np.hstack([starts[0], endings[0] ]), np.hstack([  starts[1] , endings[1] ]), c = colors[dim_c] , alpha = 0.5, s = 10) 
            
            
            # ###########################################
            # # CENTER
            # ###########################################
            # xs =  endings[0] #np.hstack([starts[0], endings[0] ])
            # mean_x = np.mean(xs)
            # ys = endings[1] #np.hstack([starts[1], endings[1] ])
            # mean_y =  np.mean(ys)
            # fig2, ax2 = plt.subplots()
            # mean_x_y = (xs - mean_x)**2 + (ys -mean_y)**2
            # ax2.hist(mean_x_y)
            
            
            
            if plot_curves:            
                [ax.plot([starts[0][i], endings[0][i]],[starts[1][i], endings[1][i]], 
                         c = colors[dim_c] , alpha = 0.5) 
                 for i in range(starts.shape[1])]
            

            
            [ax.fill([starts[0,i], starts[0,i+1], endings[0,i+1], endings[0,i]], [starts[1,i], starts[1,i+1], endings[1,i+1], endings[1,i]], 
                       color = colors[dim_c], alpha = 1) for i in np.arange(endings.shape[1]-1)]
            if closed_loop:
                ax.fill([starts[0,-1], starts[0,0], endings[0,0], endings[0,-1]], [starts[1,-1], starts[1,0], endings[1,0], endings[1,-1]], 
                           color = colors[dim_c], alpha = 1) 
                
            
            x = endings[0]
            y = endings[1]
            
            remove_edges(ax)
    elif mat.shape[0] == 3:
        pass
        
    else:
            
        raise ValueError('cannot plot high dims')
        
        

def remove_edges(ax, include_ticks = False, top = False, right = False, bottom = False, left = False):
    """
    Remove edges and ticks from a matplotlib axes.
    
    Parameters:
    - ax (matplotlib.axes.Axes): Axes from which edges and ticks will be removed.
    - include_ticks (bool): If True, ticks will be removed as well.
    - top (bool): If True, remove the top edge.
    - right (bool): If True, remove the right edge.
    - bottom (bool): If True, remove the bottom edge.
    - left (bool): If True, remove the left edge.
    """
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])            
        
        
        
        

def gaussian_vals(mat, std = 1, mean = 0 , norm = False, dimensions = 1, mat2 = [], power = 2):
    """
    check_again
    Parameters
    ----------
    mat : the matrix to consider
    std : number, gaussian std
    mean : number, optionalis 
        mean gaussian value. The default is 0.
    norm : boolean, optional
        whether to divide values by sum (s.t. sum -> 1). The default is False.

    Returns
    -------
    g : gaussian values of mat

    """    
    if dimensions == 1:
        if not checkEmptyList(mat2): 
            warnings.warn('Pay attention that the calculated Gaussian is 1D. Please change the input "dimensions" in "gaussian_vals" to 2 if you want to consider the 2nd mat as well')
      

        g = np.exp(-((mat-mean)/std)**power)
        if norm: return g/np.sum(g)

    elif dimensions == 2:
        #dim1_mat = np.abs(mat1.reshape((-1,1)) @ np.ones((1,len(mat1.flatten()))))
        #dim2_mat = np.abs((mat2.reshape((-1,1)) @ np.ones((1,len(mat2.flatten())))).T)
        #g= np.exp(-0.5 * (1/std)* (dim1_mat**power + (dim1_mat.T)**power))
        g = gaussian_vals(mat, std , mean , norm , dimensions = 1, mat2 = [], power = power)
        g1= g.reshape((1,-1))
        g2 = np.exp(-0.5/np.max([int(len((mat2-1)/2)),1])) * mat2.reshape((-1,1))
        g = g2 @ g1 
        
        g[int(g.shape[0]/2), int(g.shape[1]/2)] = 0
        if norm:
            g = g/np.sum(g)
        
    else:
        raise ValueError('Invalid "dimensions" input')
    return g
        
def cut_gauss(gaussian, t, wind, left, right):
    """
    Cuts a Gaussian array to fit within specified left and right boundaries.
    
    Parameters:
        gaussian (numpy.ndarray): The 1D Gaussian array to be cut.
        t (int): The center index around which the Gaussian array is considered.
        wind (int): The half-size of the window around the center index 't'.
        left (int): The left boundary index of the desired region.
        right (int): The right boundary index of the desired region.
    
    Returns:
        numpy.ndarray: The trimmed Gaussian array that fits within the specified boundaries.
    """
    if t + wind > right:
        diff = t + wind - right
        return gaussian[:-diff]
    elif t - wind < left:
        diff = left - (t - wind)
        return gaussian[diff:]
    else:
        return gaussian

def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1):
    """
    Convolve a 2D matrix with a Gaussian kernel along the specified direction.
    
    Parameters:
        mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
        wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
        direction (int, optional): The direction of convolution. 
            1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.
    
    Returns:
        numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.
        
    Raises:
        ValueError: If 'direction' is not 0 or 1.
    """
    if direction == 1:
        gaussian = gaussian_array(2*wind,sigma)
        mat_shape = mat.shape[1]
        return np.vstack([ [np.sum(mat[row, np.max([t - wind,0]): np.min([t + wind, mat_shape])]*cut_gauss(gaussian, t, wind, left = 0, right = mat_shape)) 
                     for t in range(mat.shape[1])] 
                   for row in range(mat.shape[0])])
    elif direction == 0:
        return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
    else:
        raise ValueError('invalid direction')
    
    
    
    
def gaussian_array(length,sigma = 1  ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
    
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    normalized_gaussian = gaussian / np.max(gaussian)
    return normalized_gaussian
    


def create_flower( stem_color = 'green', fig = [], ax = [],                 
    F = 509,
    w = 0.002,
    l = 0.4, center = (0,0), seed = 0, cos_max_c = 10, cos_max_c2 = 5, w_c_2 = 5, w_c = 8, fill_inside = True):
    
    
    
    
    x_green = [-w + center[0], w+ center[0], w+ center[0], -w+ center[0]]
    rad = 0.05
    y_green = [-rad + center[1], -rad + center[1], -rad + center[1] - l, -rad + center[1] -l]
    mat = np.vstack([0.05*np.sin(np.linspace(0,np.pi*2,F)),0.05*np.cos(np.linspace(0,np.pi*2,F))]) + np.array(center).reshape((-1,1))
    
    ax.fill(x_green,y_green, color ='green')
    c = np.vstack([w_c*np.sin(np.pi*np.linspace(0,cos_max_c, F)),w_c_2*np.sin(np.pi*np.linspace(0,cos_max_c2, F))]) + 2
    if checkEmptyList(ax):
        fig, ax = plt.subplots(figsize = (7,8))
    to_create_plot_of_cs(c, mat, fig = fig, ax = ax, seed = seed)
    if fill_inside:
        ax.fill(mat[0], mat[1], color = 'lightpink', alpha = 0.6)
    
    
    
    
def create_flowers(num_flowers = 170, F_max = 500, F_min = 60,
    w_min = 0.002, w_max = 0.02, l_max = 0.4, l_min = 0.2, 
    centers_min = -3, centers_max = 3, to_save = False, centers = [], fig = [], ax = [], name_save = ''):
    
    if checkEmptyList(centers):
        centers_min = -int(num_flowers/11)
        centers_min = int(num_flowers/11) 
        centers = np.random.rand(num_flowers, 2 )*(- centers_min + centers_max) + centers_min
    if checkEmptyList(ax):    
        fig, ax = plt.subplots(figsize = (22,5))    
    random_rb = np.random.rand( num_flowers, 2)*0.3
    greens = np.hstack([np.ones((num_flowers, 1)), random_rb])
    
    
    num_dots = np.random.choice(np.arange(F_min, F_max), size = num_flowers)
    ws = np.random.rand(num_flowers)*(w_min - w_max) + w_min 
    ls = np.random.rand(num_flowers)*(-l_min +  l_max) + l_min 
   
    cos_max_cs = np.random.randint(2,6, size = num_flowers)
    cos_max_c2s = np.random.randint(4,10, size = num_flowers)
    w_c_2s = np.random.randint(4, 8, size = num_flowers)
    w_c_s = np.random.randint(2,12, size = num_flowers)
    for flower in range(num_flowers):
        cur_stem_color = greens[flower, :]
        cur_center = centers[flower,:]

        w_current = ws[flower] 
        l = ls[flower]     
        cos_max_c = cos_max_cs[flower]
        cos_max_c2 = cos_max_c2s[flower]
        w_c_2 = w_c_2s[flower]
        w_c = w_c_s[flower]
            
        create_flower(stem_color = cur_stem_color, fig = fig, ax = ax,  center = cur_center,               
            F = num_dots[flower],
            w = w_current,
            l = l, seed = flower, cos_max_c = cos_max_c, cos_max_c2 = cos_max_c2, w_c_2 = w_c_2, w_c = w_c
            )


    if to_save:
        fig.tight_layout()
        plt.savefig('flowers_%d%s.png'%(num_flowers, name_save))
        plt.savefig('flowers_%d%s.svg'%(num_flowers, name_save))
        plt.close()
    
    
    
    
    
def create_text_to_image(string = 'BRING THEM HOME!'):    
    fig, ax = plt.subplots(figsize = (15,1))
    ax.text(0, 0, string, fontsize = 50)
    ax.set_ylim([-0.1,0.1])
    fig.tight_layout()
    
    remove_edges(ax)
    plt.savefig('%s.png'%string, bbox_inches="tight")
    
import time    
from PIL import Image    
import os 

def load_and_convert_to_bw(file_path):
    # Open the image file
    image = Image.open(file_path)

    # Convert the image to grayscale
    image_bw = image.convert('L')

    # Convert the grayscale image to a NumPy array
    image_array = np.array(image_bw)

    return image_array
    
    
def find_text_locs(string):
    if not os.path.exists('%s.png'%string):
        create_text_to_image(string)
        time.sleep(5)  # Pause for 5 seconds
    img_array = load_and_convert_to_bw('%s.png'%string)
    is_text = img_array < 250
    #print(np.where(is_text))
    text_rows, text_cols  = np.where(is_text) # np.unravel_index(np.where(is_text)[0], img_array.shape )
    
    return text_rows, text_cols
    
    
    
def flowers_by_text(string ='BRING THEM HOME', num_flowers = 1030, F_max = 60, F_min = 20,
    w_min = 0.002, w_max = 0.02, l_max = 0.4, l_min = 0.2, 
    centers_min = -3, centers_max = 3, to_save = True, fac = 0.1):
    max_flowers = num_flowers

    text_rows, text_cols = find_text_locs(string)
    text_rows = - text_rows
    if len(text_rows) > max_flowers:
        num_cur = len(text_rows)
        num_new = np.random.choice( np.arange( num_cur),  max_flowers  ) #np.linspace(0, num_cur - 1, max_flowers ).astype(int)

        text_rows_new = text_rows[num_new]
        #print(text_rows_new )
        #print(5454)
        #input('fjkfld?')
        text_cols_new = text_cols[num_new]
    else:
        num_flowers = len(text_rows) 
    

    
    fig, ax = plt.subplots(figsize = (22,5))    
    ax.scatter(text_cols*fac ,text_rows*fac, alpha = 0.2, color = 'green')
    
    
    create_flowers(num_flowers, F_max, F_min ,
        w_min, w_max, l_max, l_min, 
        centers_min, centers_max, to_save, centers = np.vstack([text_cols_new ,text_rows_new ]).T*fac, fig = fig, ax = ax, name_save = string)




    
    
    
to_run = False    


if to_run:    
    
    
    F = 509
    w = 0.002
    l = 0.4
    x_green = [-w, w, w, -w]
    y_green = [-0.2, -0.2, -0.2 - l, -0.2-l]
    
    
    mat = np.vstack([0.1*np.sin(np.linspace(0,np.pi*2,F)),0.2*np.cos(np.linspace(0,np.pi*2,F))])
    c = np.vstack([np.sin(np.pi*np.linspace(0,10, F)),5*np.cos(np.pi*np.linspace(0,8, F))]) + 5
    
    fig, ax = plt.subplots(figsize = (7,8))
    to_create_plot_of_cs(c,mat, fig = fig, ax = ax)
    ax.fill(x_green,y_green, color ='green')
    
    
    
    
    mat = np.vstack([0.1*np.sin(np.linspace(0,np.pi*2,F)),0.2*np.cos(np.linspace(0,np.pi*2,F))])
    c = np.vstack([np.sin(np.pi*np.linspace(0,10, F)),5*np.cos(np.pi*np.linspace(0,8, F))]) + 2
    
    fig, ax = plt.subplots(figsize = (7,8))
    to_create_plot_of_cs(c,mat, fig = fig, ax = ax)
    ax.fill(x_green,y_green, color ='green')
        
    mat = np.vstack([0.1*np.sin(np.linspace(0,np.pi*2,F)),0.2*np.cos(np.linspace(0,np.pi*2,F))])
    c = np.vstack([np.sin(np.pi*np.linspace(0,20, F)),5*np.sin(np.pi*np.linspace(0,18, F))]) + 2
    
    fig, ax = plt.subplots(figsize = (7,8))
    to_create_plot_of_cs(c,mat, fig = fig, ax = ax)
    ax.fill(x_green,y_green, color ='green')
        
        