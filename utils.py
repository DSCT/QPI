import numpy as np
import skimage
from scipy.ndimage import zoom, measurements
from scipy.signal import find_peaks
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colorbar import Colorbar
from PIL import Image


N=128
cmap = plt.cm.jet


## Function for crop the object

def importRI(filePath):    
    RI = np.fromfile(filePath, dtype=np.float32).reshape((512,512,512))
    RI = np.transpose(RI, (2, 1, 0))
#     RI = RI[::-1,:,:]
    return RI


def cropRI(image):
    if(len(np.shape(image)) is 3):
        (nx, ny, nz) = np.shape(image)
        if nx == ny and ny == nz:
            
            # get binary image and center of mass
            thresh = skimage.filters.threshold_otsu(image)
            binary = image > thresh    #elimating small objects might be helpful
            center_of_mass = measurements.center_of_mass(binary)
            
            cy, cx, cz = center_of_mass        
#             fig, ax = plt.subplots()
#             im = ax.imshow(np.sum(binary, axis = 2))
#             ax.plot(cx, cy, 'x')
#             plt.show()

            #recrop the RI
            x1 = (int)(cx - N//2)
            if x1 < 0:
                x1 = 0
            x2 = x1 + N
            if x2>=nx-1:
                x2 = nx - 1
                x1 = x2 - N
            
            y1 = (int)(cy - N//2)
            if y1 < 0:
                y1 = 0
            y2 = y1 + N
            if y2>=ny-1:
                y2 = ny - 1
                y1 = y2 - N
            
            z1 = (int)(cz - N//2)
            if z1 < 0:
                z1 = 0
            z2 = z1 + N
            if z2>=nz-1:
                z2 = nz - 1
                z1 = z2 - N
                
            crop_img = image[y1:y2, x1:x2, z1:z2]
            cx = int(cx - x1)
            cy = int(cy - y1)
            cz = int(cz - z1)
            
            return crop_img,(cx, cy, cz)
    else:
        return np.ones((N,N,N)), (0,0,0)
    
    
    
def resolution(data, idx1, idx2, dx):
    dist = data[idx1:idx2]
    if dist[0] > dist[-1]:
        dist = np.flip(dist)
    
    ri10 = dist[0] + (dist[-1] - dist[0])*0.1
    ri90 = dist[0] + (dist[-1] - dist[0])*0.9
    
    ri10L = np.where(ri10-dist > 0)[0][-1]
    ri10R = np.where(ri10-dist < 0)[0][0]
    
    ri90L = np.where(ri90-dist > 0)[0][-1]
    ri90R = np.where(ri90-dist < 0)[0][0]
    
    p_ri10 = ri10L + (ri10 - dist[ri10L]) / (dist[ri10R] - dist[ri10L]) * (ri10R - ri10L)
    p_ri90 = ri90L + (ri90 - dist[ri90L]) / (dist[ri90R] - dist[ri90L]) * (ri90R - ri90L)
    
    return (p_ri90 - p_ri10) * dx, (p_ri10+idx1, p_ri90+idx1)




## Function for calculating the resolution

def analysis(data, dx, high=1.3336):

    max_peaks, _ = find_peaks(data, height=high)
    min_peaks, _ = find_peaks(data.max() - data, height=-data.max())
    peaks = np.sort(np.concatenate((max_peaks, min_peaks)))
    left_idx = np.sort(data[max_peaks])[-1]
    right_idx = np.sort(data[min_peaks])[-1]

    max_value = np.sort(data[max_peaks])
    min_value = data[min_peaks]
    values = data[peaks]
    dV = np.diff(values)
    
    if len(max_peaks) >=2 and len(max_peaks)>0: 
        L2 = np.where(data == max_value[-1])[0][0]
        R1 = np.where(data == max_value[-2])[0][0]

        if L2>R1:
            tmp = L2
            L2 = R1
            R1 = tmp        
    else:
        L2 = np.where(data == max_value[-1])[0][0]
        R1=L2
        
    dL = np.diff(data[:L2])
    dR = np.diff(data[R1:])
    
    L1 = np.where(dL == 0)[0][-1]+1
    R2 = np.where(dR == 0)[0][0]+R1+1
    
    res1, (L10_1,L10_2) = resolution(data, L1, L2, dx)
    res2, (R10_1,R10_2) = resolution(data, R1, R2, dx)
    
#     plt.plot(data)
#     plt.plot(max_peaks, data[max_peaks], "x")
#     plt.plot(min_peaks, data[min_peaks], "x")
#     plt.show()
    
    return np.min([res1,res2]), (L10_1,L10_2), (R10_1,R10_2)





## Generate the figure

def drawFig(ri_path, vs=0.21855, threshold=1.3336):
    
    ri_map = importRI(ri_path)
    crop, (cx,cy,cz) = cropRI(ri_map)

    x_sec = crop[:,cx,:]
    y_sec = crop[cy,:,:]
    z_sec = crop[:,:,cz]
    
    # get the images and profilers
    yz = np.transpose(x_sec)
    xy = np.transpose(z_sec)

    nz = np.shape(yz)[1]
    ny = np.shape(xy)[0]
    nx = np.shape(xy)[1]

    z_data = yz[:,ny//2]
    y_data = xy[:,nx//2]
    x_data = xy[ny//2,:]

    rimax = np.max([xy,yz]) + 0.005
    rimin = np.min([xy,yz]) - 0.005

    # estimate the resolution
    resZ, (zL1,zL2), (zR1,zR2) = analysis(z_data, vs, threshold+0.03)
    resY, (yL1,yL2), (yR1,yR2) = analysis(y_data, vs, threshold+0.03)
    resX, (xL1,xL2), (xR1,xR2) = analysis(x_data, vs, threshold+0.03)

    fig = plt.figure( figsize=(8, 14))

    gs = gridspec.GridSpec(3, 2, width_ratios=[6, 2], height_ratios=[6, 6, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    ax6.axis('off')

    im2=ax1.imshow(yz, cmap, extent=(-nx*vs/2, nx*vs/2, -nz*vs/2, nz*vs/2), vmin=rimin, vmax=rimax)
    im1=ax3.imshow(xy, cmap, extent=(-nx*vs/2, nx*vs/2, -ny*vs/2, ny*vs/2), vmin=rimin, vmax=rimax)
    ax2.plot(z_data,np.arange(nz)*vs, color='b', linestyle='-.')
    ax4.plot(y_data,np.arange(ny)*vs, color='b', linestyle='-.')
    ax5.plot(np.arange(nx)*vs,x_data, color='b', linestyle='-.')


    # ax1.xaxis.set_ticks_position('top')
    # ax2.xaxis.set_ticks_position('top')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    ax1.tick_params(axis='x', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)
    ax3.tick_params(axis='x', which='both', length=0)
    ax4.tick_params(axis='y', which='both', length=0)

    ax2.axis([rimin, rimax, 0, nz*vs])
    ax4.axis([rimin, rimax, 0, ny*vs])
    ax5.axis([0, ny*vs, rimin, rimax])

    ax1.plot([-nx*vs/2, nx*vs/2], [0,0], color='r', linestyle='-', linewidth=0.5)  
    ax1.plot([0,0], [-nz*vs/2, nz*vs/2], color='r', linestyle='-', linewidth=0.5) 
    ax1.plot([-2.5,2.5], [-nx*vs/2+1.5,-nx*vs/2+1.5], color='w', linestyle='-', linewidth=5)   
    ax1.arrow(-nx*vs/2+1.5, -nz*vs/2+1.5, 0,3, width=0.4, shape="left", color='w')
    ax1.arrow(-nx*vs/2+1.5, -nz*vs/2+1.5, 3,0, width=0.4, shape="right", color='w')

    ax3.plot([-nx*vs/2, nx*vs/2], [0,0], color='r', linestyle='-', linewidth=0.5)  
    ax3.plot([0,0], [-nx*vs/2, nx*vs/2], color='r', linestyle='-', linewidth=0.5) 
    ax3.plot([-2.5,2.5], [-nx*vs/2+1.5,-nx*vs/2+1.5], color='w', linestyle='-', linewidth=5)  
    ax3.arrow(-nx*vs/2+1.5, ny*vs/2-1.5, 0,-3, width=0.4, shape="right", color='w')
    ax3.arrow(-nx*vs/2+1.5, ny*vs/2-1.5, 3,0, width=0.4, shape="left", color='w')

    # ax1.grid(True)
    # ax2.grid(True)
    # ax3.grid(True)
    # ax4.grid(True)
    # ax5.grid(True)

    # plot the lines of profilers at 10% and 90%
    ax2.text(0.5, 1, 'z-axis', verticalalignment='top', horizontalalignment='center', transform=ax2.transAxes, color='black', fontsize=15)
    ax2.plot([rimin, rimax], [zL1*vs, zL1*vs], color='r', linestyle='-', linewidth=0.5)
    ax2.plot([rimin, rimax], [zL2*vs, zL2*vs], color='r', linestyle='-', linewidth=0.5)
    ax2.plot([rimin, rimax], [zR1*vs, zR1*vs], color='r', linestyle='-', linewidth=0.5)
    ax2.plot([rimin, rimax], [zR2*vs, zR2*vs], color='r', linestyle='-', linewidth=0.5)
    
    ax4.text(0.5, 1, 'y-axis', verticalalignment='top', horizontalalignment='center', transform=ax4.transAxes, color='black', fontsize=15)
    ax4.plot([rimin, rimax], [yL1*vs, yL1*vs], color='r', linestyle='-', linewidth=0.5)
    ax4.plot([rimin, rimax], [yL2*vs, yL2*vs], color='r', linestyle='-', linewidth=0.5)
    ax4.plot([rimin, rimax], [yR1*vs, yR1*vs], color='r', linestyle='-', linewidth=0.5)
    ax4.plot([rimin, rimax], [yR2*vs, yR2*vs], color='r', linestyle='-', linewidth=0.5)

    ax5.text(0.5, 1, 'x-axis', verticalalignment='top', horizontalalignment='center', transform=ax5.transAxes, color='black', fontsize=15)
    ax5.plot([xL1*vs, xL1*vs], [rimin, rimax], color='r', linestyle='-', linewidth=0.5)
    ax5.plot([xL2*vs, xL2*vs], [rimin, rimax], color='r', linestyle='-', linewidth=0.5)
    ax5.plot([xR1*vs, xR1*vs], [rimin, rimax], color='r', linestyle='-', linewidth=0.5)
    ax5.plot([xR2*vs, xR2*vs], [rimin, rimax], color='r', linestyle='-', linewidth=0.5)

    print('Resolution of x-axis: %.3f (um)' %(resX))
    print('Resolution of y-axis: %.3f (um)' %(resY))
    print('Resolution of z-axis: %.3f (um)' %(resZ))

    # modify the space of figure
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.025, wspace = 0.01)


    cbar_ax = fig.add_axes([-0.11, 0.38, 0.020, 0.4])    #L, B, W, H
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical').set_label(label='Refractive Index',size=14)
    cbar_ax.yaxis.set_ticks_position('left')

    plt.show()  
