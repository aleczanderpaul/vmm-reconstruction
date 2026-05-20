import uproot
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import crystalball


'''this file is for useful functions for analyzing data from the VMMs'''

#this opens the ROOT file and returns the hits as a Pandas dataframe
def read_hit(file_loc):
    
    file = uproot.open(file_loc)
    hits = file['hits']['hits']

    dict = {
        'id' : hits['id'].array(),
        'det' : hits['det'].array(),
        'plane' : hits['plane'].array(),
        'fec' : hits['fec'].array(),
        'vmm' : hits['vmm'].array(),
        'readout_time' : hits['readout_time'].array(),
        'time' : hits['time'].array(),
        'ch' : hits['ch'].array(),
        'pos' : hits['pos'].array(),
        'bcid' : hits['bcid'].array(),
        'tdc' : hits['tdc'].array(),
        'adc' : hits['adc'].array(),
        'over_threshold' : hits['over_threshold'].array(),
        'chip_time' : hits['chip_time'].array()
        }

    df = pd.DataFrame(data = dict)

    return df

#this opens the ROOT file and returns the clusters as a Pandas dataframe
def read_cluster(file_loc):

    file = uproot.open(file_loc)

    clusters_detector = file['clusters_detector']['clusters_detector']

    dict = {
    'id' : clusters_detector['id'].array(),
    'id0' : clusters_detector['id0'].array(),
    'id1' : clusters_detector['id1'].array(),
    'id2' : clusters_detector['id2'].array(),
    'det' : clusters_detector['det'].array(),
    'size0' : clusters_detector['size0'].array(),
    'size1' : clusters_detector['size1'].array(),
    'size2' : clusters_detector['size2'].array(),
    'adc0' : clusters_detector['adc0'].array(),
    'adc1' : clusters_detector['adc1'].array(),
    'adc2' : clusters_detector['adc2'].array(),
    'pos0' : clusters_detector['pos0'].array(),
    'pos1' : clusters_detector['pos1'].array(),
    'pos2' : clusters_detector['pos2'].array(),
    'time0' : clusters_detector['time0'].array(),
    'time1' : clusters_detector['time1'].array(),
    'time2' : clusters_detector['time2'].array(),
    'pos0_utpc' : clusters_detector['pos0_utpc'].array(),
    'pos1_utpc' : clusters_detector['pos1_utpc'].array(),
    'pos2_utpc' : clusters_detector['pos2_utpc'].array(),
    'time0_utpc' : clusters_detector['time0_utpc'].array(),
    'time1_utpc' : clusters_detector['time1_utpc'].array(),
    'time2_utpc' : clusters_detector['time2_utpc'].array(),
    'pos0_charge2' : clusters_detector['pos0_charge2'].array(),
    'pos1_charge2' : clusters_detector['pos1_charge2'].array(),
    'pos2_charge2' : clusters_detector['pos2_charge2'].array(),
    'time0_charge2' : clusters_detector['time0_charge2'].array(),
    'time1_charge2' : clusters_detector['time1_charge2'].array(),
    'time2_charge2' : clusters_detector['time2_charge2'].array(),
    'pos0_algo' : clusters_detector['pos0_algo'].array(),
    'pos1_algo' : clusters_detector['pos1_algo'].array(),
    'pos2_algo' : clusters_detector['pos2_algo'].array(),
    'time0_algo' : clusters_detector['time0_algo'].array(),
    'time1_algo' : clusters_detector['time1_algo'].array(),
    'time2_algo' : clusters_detector['time2_algo'].array(),
    'dt0' : clusters_detector['dt0'].array(),
    'dt1' : clusters_detector['dt1'].array(),
    'dt2' : clusters_detector['dt2'].array(),
    'delta_plane_0_1' : clusters_detector['delta_plane_0_1'].array(),
    'delta_plane_1_2' : clusters_detector['delta_plane_1_2'].array(),
    'delta_plane_0_2' : clusters_detector['delta_plane_0_2'].array(),
    'span_cluster0' : clusters_detector['span_cluster0'].array(),
    'span_cluster1' : clusters_detector['span_cluster1'].array(),
    'span_cluster2' : clusters_detector['span_cluster2'].array(),
    'max_delta_time0' : clusters_detector['max_delta_time0'].array(),
    'max_delta_time1' : clusters_detector['max_delta_time1'].array(),
    'max_delta_time2' : clusters_detector['max_delta_time2'].array(),
    'max_missing_strip0' : clusters_detector['max_missing_strip0'].array(),
    'max_missing_strip1' : clusters_detector['max_missing_strip1'].array(),
    'max_missing_strip2' : clusters_detector['max_missing_strip2'].array(),
    }

    df = pd.DataFrame(data = dict)

    return df

#combine the hit and cluster data of every ROOT file in a folder and return Pandas dataframes
def combineDataFrames(rootFolder): #input is string with the name of the folder
    rootFiles = sorted(glob.glob(os.path.join(rootFolder, "*.root"))) #using the sorted feature assuming the filenames have a meaning (e.g., chronological)
    hits = []
    clusters = []
    data_duration = 0.0
    for filePath in rootFiles:
        df_hits = read_hit(filePath)
        hits.append(df_hits)
        data_duration += (df_hits['time'].max() - df_hits['time'].min()) * (1e-9)

        clusters.append(read_cluster(filePath))

    df_hits = pd.concat(hits, ignore_index=True)
    df_clusters = pd.concat(clusters, ignore_index=True)
    return df_hits, df_clusters, data_duration

# Fit a Crystal Ball function to fe55 events
def fitCB(df, plot=True, saveFig=True):

    try:
        # Get gain values
        gain = df.gain
        # Keep only gain entries with z-score < 3 (exclude outlier which may be cosmic tracks or nuclear recoils)
        gain =  gain[(np.abs(stats.zscore(gain)) < 3)]

        # Do not attempt fit if there are less then 100 examples
        if len(gain) < 100:
            raise Exception("Poor fit")

        xmin = 0
        xmax = gain.max()
        nbins = 100

        hist, bin_edges = np.histogram(gain,nbins,(xmin,xmax))
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.
        # Find Non-zero bins in Histogram
        nz = hist>0
        # Get error bars for bins
        n_err = np.sqrt(hist[nz])

        # Create numpy Histogram, use density this time
        hist2, bin_edges2 = np.histogram(gain,nbins,(xmin,xmax), density = True)
        bin_centers2 = (bin_edges2[1:]+bin_edges2[:-1])/2.

        # Guess mu as bin_center with most hits
        mu_guess = bin_centers2[np.argmax(hist2)]

        # Find Non-zero bins in Histogram
        nz2 = hist2>0
        # Get error bars for bins
        n_err2 = (np.sqrt(hist[nz])/hist[nz]) * hist2[nz2] # Fractional error times hist value

        # Define Range and Fit :
        try:
            coeff, covar = curve_fit(crystalball.pdf, bin_centers2[nz2], hist2[nz2], sigma=n_err2, absolute_sigma=True, p0=(1, 2,mu_guess,1600)) #p0 = beta, m, mu, sigma
        except:
            coeff, covar = curve_fit(crystalball.pdf, bin_centers2[nz2], hist2[nz2], sigma=n_err2, absolute_sigma=True, p0=(2, 2,mu_guess,1600))


        f_opti = crystalball.pdf(bin_centers,*coeff)

        perr = np.sqrt(np.diag(covar))

        if np.absolute(perr[2]) > np.absolute(coeff[2]):
            raise Exception("Poor fit")


        if plot == True and saveFig == True:
            plt.figure()
            hist2, bin_edges2, patches2 = plt.hist(gain,nbins,(xmin,xmax), density = True, color='g',alpha=0.6)
            bin_centers2 = (bin_edges2[1:]+bin_edges2[:-1])/2.
            plt.xlabel("Gain")
            plt.ylabel("Probability Density")
            plt.plot(bin_centers, f_opti, 'r--', linewidth=2, label='CB Fit')
            plt.legend(loc='upper right')
            plt.savefig('gain_fit_CB.png', bbox_inches="tight")
            plt.close()
        elif plot == True and saveFig == False: #if saveFig is False, user will need to add plt.figure() before calling this function
            hist2, bin_edges2, patches2 = plt.hist(gain,nbins,(xmin,xmax), density = True, color='g',alpha=0.6)
            bin_centers2 = (bin_edges2[1:]+bin_edges2[:-1])/2.
            plt.xlabel("Gain")
            plt.ylabel("Probability Density")
            plt.plot(bin_centers, f_opti, 'r--', linewidth=2, label='CB Fit')

        charge_sharing = 1.0*np.mean(df.electrons_x/df.electrons_y)
        mu_e_x = np.mean(df.electrons_x)
        mu_e_y = np.mean(df.electrons_y)


        return coeff[0], perr[0], coeff[1], perr[1], coeff[2], perr[2], coeff[3], perr[3], charge_sharing, mu_e_x, mu_e_y

    except:
        print("-fit failed-")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# Fiducializes dataframe so that clusters are contained in a specified area
def fiducializeArea(df_cluster, area):

    #this is based on the Zander_setup geometry file (VMM 8-15 on x, VMM 0-7 on y)
    if area == 'd': #VMM 10, 2
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 156)  & (df_cluster.pos0 <= 217) & (df_cluster.pos1 >= 280) & (df_cluster.pos1 <=  342)].reset_index()
    elif area == 'c': #VMM 10, 5
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 156)  & (df_cluster.pos0 <= 217) & (df_cluster.pos1 >= 156) & (df_cluster.pos1 <=  217)].reset_index()
    elif area == 'b': #VMM 13, 2
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 280)  & (df_cluster.pos0 <= 342) & (df_cluster.pos1 >= 280) & (df_cluster.pos1 <=  342)].reset_index()
    elif area == 'a': #VMM 13, 5
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 280)  & (df_cluster.pos0 <= 342) & (df_cluster.pos1 >= 156) & (df_cluster.pos1 <=  217)].reset_index()
    elif area == 'bottom right': 
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 280) & (df_cluster.pos1 <=  217)].reset_index()
    elif area == 'bottom left': 
        df_fid = df_cluster.loc[(df_cluster.pos0 <= 217) & (df_cluster.pos1 <=  217)].reset_index()
    elif area == 'sensitive': 
        df_fid = df_cluster.loc[(df_cluster.pos0 >= 253) & (df_cluster.pos0 <=  260) & (df_cluster.pos1 >= 242) & (df_cluster.pos1 <=  269)].reset_index()
    else:
        raise Exception("provide valid area")

    return df_fid

#this opens the ROOT file and returns the clusters as a Pandas dataframe, works only for Majd's data
def read_cluster_Majd(file_loc):

    file = uproot.open(file_loc)

    clusters_detector = file['clusters_detector']['clusters_detector']

    dict = {
    'id' : clusters_detector['id'].array(),
    'id0' : clusters_detector['id0'].array(),
    'id1' : clusters_detector['id1'].array(),
    'det' : clusters_detector['det'].array(),
    'size0' : clusters_detector['size0'].array(),
    'size1' : clusters_detector['size1'].array(),
    'adc0' : clusters_detector['adc0'].array(),
    'adc1' : clusters_detector['adc1'].array(),
    'pos0' : clusters_detector['pos0'].array(),
    'pos1' : clusters_detector['pos1'].array(),
    'time0' : clusters_detector['time0'].array(),
    'time1' : clusters_detector['time1'].array(),
    'pos0_utpc' : clusters_detector['pos0_utpc'].array(),
    'pos1_utpc' : clusters_detector['pos1_utpc'].array(),
    'time0_utpc' : clusters_detector['time0_utpc'].array(),
    'time1_utpc' : clusters_detector['time1_utpc'].array(),
    'pos0_charge2' : clusters_detector['pos0_charge2'].array(),
    'pos1_charge2' : clusters_detector['pos1_charge2'].array(),
    'time0_charge2' : clusters_detector['time0_charge2'].array(),
    'time1_charge2' : clusters_detector['time1_charge2'].array(),
    'pos0_algo' : clusters_detector['pos0_algo'].array(),
    'pos1_algo' : clusters_detector['pos1_algo'].array(),
    'time0_algo' : clusters_detector['time0_algo'].array(),
    'time1_algo' : clusters_detector['time1_algo'].array(),
    'dt0' : clusters_detector['dt0'].array(),
    'dt1' : clusters_detector['dt1'].array(),
    'delta_plane' : clusters_detector['delta_plane'].array(),
    'span_cluster0' : clusters_detector['span_cluster0'].array(),
    'span_cluster1' : clusters_detector['span_cluster1'].array(),
    'max_delta_time0' : clusters_detector['max_delta_time0'].array(),
    'max_delta_time1' : clusters_detector['max_delta_time1'].array(),
    'max_missing_strip0' : clusters_detector['max_missing_strip0'].array(),
    'max_missing_strip1' : clusters_detector['max_missing_strip1'].array(),
    }

    df = pd.DataFrame(data = dict)

    return df

#combine the hit and cluster data of every ROOT file in a folder and return Pandas dataframes, works only for Majd's data
def combineDataFramesMajd(rootFolder): #input is string with the name of the folder
    rootFiles = sorted(glob.glob(os.path.join(rootFolder, "*.root"))) #using the sorted feature assuming the filenames have a meaning (e.g., chronological)
    hits = []
    clusters = []
    for filePath in rootFiles:
        hits.append(read_hit(filePath))
        clusters.append(read_cluster_Majd(filePath))

    df_hits = pd.concat(hits, ignore_index=True)
    df_clusters = pd.concat(clusters, ignore_index=True)
    return df_hits, df_clusters

def read_hits_in_cluster(file_loc):

    file = uproot.open(file_loc)

    clusters_detector = file['clusters_detector']['clusters_detector']

    strips0 = clusters_detector['strips0'].array()
    strips1 = clusters_detector['strips1'].array()
    strips2 = clusters_detector['strips2'].array()
    adcs0 = clusters_detector['adcs0'].array()
    adcs1 = clusters_detector['adcs1'].array()
    adcs2 = clusters_detector['adcs2'].array()
    times0 = clusters_detector['times0'].array()
    times1 = clusters_detector['times1'].array()
    times2 = clusters_detector['times2'].array()

    df = {
    'strips0': strips0,
    'strips1': strips1,
    'strips2': strips2,
    'adcs0':   adcs0,
    'adcs1':   adcs1,
    'adcs2':   adcs2,
    'times0':  times0,
    'times1':  times1,
    'times2':  times2,
    }

    return df

def read_hits_in_cluster_majd(file_loc):

    file = uproot.open(file_loc)

    clusters_detector = file['clusters_detector']['clusters_detector']

    strips0 = clusters_detector['strips0'].array()
    strips1 = clusters_detector['strips1'].array()
    adcs0 = clusters_detector['adcs0'].array()
    adcs1 = clusters_detector['adcs1'].array()
    times0 = clusters_detector['times0'].array()
    times1 = clusters_detector['times1'].array()

    df = {
    'strips0': strips0,
    'strips1': strips1,
    'adcs0':   adcs0,
    'adcs1':   adcs1,
    'times0':  times0,
    'times1':  times1,
    }

    return df

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# A function to get the transverse mismeasurments of a 3D reconstrcuted track
def GetTransErrs(x_vals,y_vals,z_vals,charges,charge_weighting = True):

    if charge_weighting == False:
        charges = np.ones(len(x_vals))

    X = np.array([x_vals,y_vals,z_vals]).T

    # 1) Center on barycenter
    # Barycenter is the charge-weighted mean position
    x_b = np.sum(X*(charges.reshape(len(charges),1)),axis=0)/np.sum(charges)
    # Shift data to barycenter
    X = X-x_b

    # 2) Find principle axis
    # Use charges for weights
    W = charges.reshape(len(charges),1)
    # Compute weighted covariance matrix
    WCM = ( (W*X).T @ X ) / np.sum(W)
    U1,S1,D1 =  np.linalg.svd(WCM)
    v_PA = np.array([D1[0][0],D1[0][1],D1[0][2]])

    v_PA = np.sign(v_PA[2]) * v_PA

    # projection of mean-centered position onto principle axis
    proj = np.array([(X@v_PA)*v_PA[0],(X@v_PA)*v_PA[1],(X@v_PA)*v_PA[2]]).T

    # Mismeasurement vectors
    # The distribution of the x and y values gives us sigma x and sigma y
    err =X-proj

    # Compute transverse mismeasurements using method 1 (see slides)
    delta_xs_1 = err[:,0]
    delta_ys_1 = err[:,1]

    # Compute transverse mismeasurements using method 2 (see slides)
    delta_xs_2 = X[:,0] - ( ( v_PA[0] / v_PA[2] ) * X[:,2] ) 
    delta_ys_2 = X[:,1] - ( ( v_PA[1] / v_PA[2] ) * X[:,2] )
    
    return z_vals, delta_xs_1, delta_ys_1, delta_xs_2, delta_ys_2, v_PA, x_b

def Reconstruction3D(mu, sigma, n_sigma, times_x, times_y, ADC_x, ADC_y, strips_x, strips_y, gain_x, gain_y, pitch_x, pitch_y, v_drift):
    # This 3D reconstruction algorithim only matches x and y hits if they are within a time window specified by mu, sigma, n_sigma
    # After x and y hits are matched, the x ADCs are spread evenly among all matched y hits and vice versa
    # The time is the average of the x hit time and y hit time
    # Unmatched hits are spread along all matched vertices via a time-weighted spread

    # Truth array - contains truth value for x and y hits that fire within the time gap window.
    # i.e. if Tarray_{ij} = True, then the ith x hit and the jth y hit are within the gap window
    # and should be combined, this constitutes an xy-hit
    Tarray = np.abs((np.subtract.outer(times_x,times_y)-mu) / sigma) < n_sigma

    if (True in Tarray) == False:
        print(f"Warning: None of the hits are matched in time within {n_sigma} sigma")


    # This counts the number of simultaniously triggering y hits for each x hit
    TCol = np.sum(Tarray,axis=1)*1.0
    # This counts the number of simultaniously triggering x hits for each y hit
    TRow = np.sum(Tarray,axis=0)*1.0

    # Throw an error if there are unmatched hits
    # This can be updated later
    if (0 in TCol) or (0 in TRow):
        print("Warning: Unmatched hits. Performing time-weighted spread")

    # Collect unmatched hit info
    # Convert ADC to electron count units
    unmatched_ADCs = np.append(ADC_x[ TCol == 0 ] * (6240.0 / gain_x), ADC_y[ TRow == 0 ] * (6240.0 / gain_y))
    # Shift x and y times based on mean offset
    unmatched_times = np.append(times_x[ TCol == 0 ] - (mu/2.0) ,  times_y[ TRow == 0 ] + (mu/2.0) )

    # Rebuild arrays, ommiting unmatched hits
    # Convert ADC to electron count units
    x_times = times_x[ TCol > 0 ]
    ADC_x = ADC_x[ TCol > 0 ] * (6240.0 / gain_x)
    strips_x = strips_x[ TCol > 0 ]
    y_times = times_y[ TRow > 0 ]
    ADC_y = ADC_y[ TRow > 0 ] * (6240.0 / gain_y)
    strips_y = strips_y[ TRow > 0 ]
    Tarray = np.abs((np.subtract.outer(x_times,y_times)-mu) / sigma) < n_sigma
    TCol = np.sum(Tarray,axis=1)*1.0
    TRow = np.sum(Tarray,axis=0)*1.0

    # This divides the ADC of the x hit by the number of simultaniously triggering y hits
    ADCx_V = np.divide(ADC_x,TCol)

    # This is a matrix of the x ADC contribution to all xy-hits
    elecx_M = np.multiply(ADCx_V[..., None],Tarray)

    # This divides the ADC of the y hit by the number of simultaniously triggering x hits
    ADCy_V = np.divide(ADC_y,TRow)

    # This is a matrix of the y ADC contribution to all xy-hits
    elecy_M = np.multiply(ADCy_V,Tarray)

    # This is the total ADC assigned to each xy-hit
    elec_M = elecx_M+elecy_M

    # This holds the x strip position for each xy-hit
    Stripx_M = np.multiply(strips_x[..., None],Tarray)

    # This holds the y strip position for each xy-hit
    Stripy_M = np.multiply(strips_y,Tarray)

    # This holds the x time measurment for each xy-hit
    Timex_M = np.multiply(x_times[..., None],Tarray)

    # This holds the y time measurment for each xy-hit
    Timey_M = np.multiply(y_times,Tarray)

    # This holds the average time measurment for each xy-hit
    Time_M = (Timex_M + Timey_M) / 2.0

    # absolute time offsets between matched vertices and unmatched hits
    abs_t_off = np.abs( Time_M-np.tensordot(unmatched_times, Tarray, axes=0) )
    # Really we want to weight by the inverse time difference 
    abs_t_off = np.reciprocal(abs_t_off, where=abs_t_off != 0, out=np.zeros_like(abs_t_off))

    # Corresponding umatched ADC and time offset normalization factor
    ADC_norm = unmatched_ADCs/abs_t_off.sum(axis=1).sum(axis=1)

    # Multiply togather and sum to get total unmatched ADC contribution for each vertex
    unmatched_contrib = (abs_t_off*np.tensordot(ADC_norm, Tarray, axes=0)).sum(axis=0)

    # Add to ADC matrix
    elec_M += unmatched_contrib

    # Convert to physical quatities
    x_vals = Stripx_M[Tarray]*pitch_x            # Multiply by pitch for physical distance
    y_vals = Stripy_M[Tarray]*pitch_y            # Multiply by pitch for physical distance
    weights = elec_M[Tarray]                     # Weight is number of electrons
    z_vals  = Time_M[Tarray]*v_drift             # multiply by drift speed for z
    z_vals = z_vals - np.min(z_vals)             # Shift z_vals so that minimum is at z=0

    return x_vals, y_vals, z_vals, weights