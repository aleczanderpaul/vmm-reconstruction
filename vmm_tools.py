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

'''    'strips0' : clusters_detector['strips0'].array(),
    'strips1' : clusters_detector['strips1'].array(),
    'strips2' : clusters_detector['strips2'].array(),
    'adcs0' : clusters_detector['adcs0'].array(),
    'adcs1' : clusters_detector['adcs1'].array(),
    'adcs2' : clusters_detector['adcs2'].array(),
    'times0' : clusters_detector['times0'].array(),
    'times1' : clusters_detector['times1'].array(),
    'times2' : clusters_detector['times2'].array(),  '''

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