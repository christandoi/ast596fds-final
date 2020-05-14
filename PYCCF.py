"""
By Mouyuan (Eric) Sun
email: ericsun88@live.com

Version history:
Version 1.0 (May 27, 2015)
Version 1.1 (Nov 15, 2015):
bug fix: make sure the peak is significant when calculate the centroid
         (i.e., add ' and status_peak==1 ' to line 212)
has since been edited fairly extensively by Kate Grier 


Cross-correlation functions
Interpolation if unevenly sampled

The idea:
Peterson et al.(1998): http://arxiv.org/abs/astro-ph/9802103
"""

import numpy as np
import scipy.stats as sst
from matplotlib import pyplot as plt



def corsig(r, v):
    '''
    Calculate the p value that a random uncorrelated sample can yield a 
    correlation coefficient as large as or larger than the observed absolute 
    value of r, where r is the correlation coefficient of the data (using 
    t test, valid if v>=4)
    Ref1: http://janda.org/c10/Lectures/topic06/L24-significanceR.htm
    Ref2: http://vassarstats.net/textbook/ch4apx.html
    
    Inputs:
        r -- the correlation coefficient of the data
        v -- degree of freedom when calculating r: N-2 (hence N>2!!!)
    Outputs:
        pvalue
    '''
    r = float(r)
    v = float(v)
    
    r2 = r*r
    if r2 == 1.:
        tst = 0.
    else:
        tst = r*np.sqrt(v/(1.-r2))
    pvalue = sst.t.sf(tst, v) # sf: survival function -- 1-CDF
    return pvalue
    



def xcor(t1, y1, t2, y2, tlagmin1, tlagmax1, tunit, imode=0):
    '''
    Calculate cross-correlation function for unevenly 
    sampling data.
    
    Inputs:
        t1 -- time for light curve 1, assume increase;
        y1 -- flux for light curve 1;
        t2 -- time for light curve 2, assume increase;
        y2 -- flux for light curve 2;
        tlagmin -- minimum time lag;
        tlagmax -- maximum time lag;
        tunit -- tau step;
        imode -- cross-correlation mode: 0, twice (default); 
                 1, interpolate light curve 1;
                 2, interpolate light curve 2.
        
    Outputs:
        ccf -- correlation coefficient;
        tlag -- time lag (t2 - t1); positive values mean second
                  light curve lags the first light curve, as per convention.
                 (edit by kate, march 2016)
        npts -- number of data points used;
    '''
    safe = tunit*0.1
    taulist1 = []
    taulist2 = []
    npts1 = []
    npts2 = []
    ccf12 = []  # interpolate 2
    ccf21 = []  # interpolate 1

    tlagmin = -tlagmax1
    tlagmax = -tlagmin1  
  
    # frist interpolate 2
    tau = tlagmin + 0.0
    while tau < tlagmax+safe:
        t2new = t1 - tau
        selin = np.where((t2new>=np.min(t2))&(t2new<=np.max(t2)), True, False)  ##orig
        #selin = np.where((t2new>=np.min(t2))&(t2new<=np.max(t2)))[0]  ##jli184
        #print selin, len(selin) ## jli184
        knot = np.sum(selin)  # number of datapoints used ##jli184
        if knot>0:
            y2new = np.interp(t2new[selin], t2, y2)
            #print len(y1)  ##jli184
            y1sum = np.sum(y1[selin])
            y1sqsum = np.sum(y1[selin]*y1[selin])
            y2sum = np.sum(y2new)
            y2sqsum = np.sum(y2new*y2new)
            y1y2sum = np.sum(y1[selin]*y2new)
            
            fn = float(knot)
            rd1_sq = fn*y2sqsum - y2sum*y2sum
            rd2_sq = fn*y1sqsum - y1sum*y1sum
            if rd1_sq>0.0:
                rd1 = np.sqrt(rd1_sq)
            else:
                rd1 = 0.0
            if rd2_sq>0.0:
                rd2 = np.sqrt(rd2_sq)
            else:
                rd2 = 0.0
            
            if rd1*rd2==0.0:
                r = 0.0
            else:
                r = (fn*y1y2sum - y2sum*y1sum)/(rd1*rd2)
            ccf12.append(r)
            taulist1.append(tau)
            npts1.append(knot)
        tau += tunit
        
    # now interpolate 1
    tau = tlagmin + 0.0
    while tau < tlagmax+safe:
        t1new = t2 + tau
        selin = np.where((t1new>=np.min(t1))&(t1new<=np.max(t1)), True, False)
        knot = np.sum(selin)  # number of datapoints used
        if knot>0:
            y1new = np.interp(t1new[selin], t1, y1)
            
            y2sum = np.sum(y2[selin])
            y2sqsum = np.sum(y2[selin]*y2[selin])
            y1sum = np.sum(y1new)
            y1sqsum = np.sum(y1new*y1new)
            y1y2sum = np.sum(y1new*y2[selin])
            
            fn = float(knot)
            rd1_sq = fn*y2sqsum - y2sum*y2sum
            rd2_sq = fn*y1sqsum - y1sum*y1sum
            if rd1_sq>0.0:
                rd1 = np.sqrt(rd1_sq)
            else:
                rd1 = 0.0
            if rd2_sq>0.0:
                rd2 = np.sqrt(rd2_sq)
            else:
                rd2 = 0.0
            
            if rd1*rd2==0.0:
                r = 0.0
            else:
                r = (fn*y1y2sum - y2sum*y1sum)/(rd1*rd2)
            ccf21.append(r)
            taulist2.append(tau)
            npts2.append(knot)
        tau += tunit

    # return results according to imode     
    if imode==0:
        if taulist1 != taulist2:
            print ('Warning: Unmatch arrays!')
            #print taulist1
            #print taulist2
            #print ccf12
            #print ccf21
            taulist = np.asarray(np.unique(taulist1+taulist2))
            ccf = np.zeros(len(taulist))
            npts = np.zeros(len(taulist))
            
            for j in range(len(taulist)):
                if taulist[j] in taulist1:
                    ind1 = taulist1.index(taulist[j])
                    ccf[j] = ccf[j]+ccf12[ind1]
                    npts[j] = npts[j]+npts1[ind1]
                    
                if taulist[j] in taulist2:
                    ind2 = taulist2.index(taulist[j])
                    ccf[j] = ccf[j]+ccf21[ind2]
                    npts[j] = npts[j]+npts2[ind2]
            

            ccf = ccf*0.5
            npts = npts*0.5
            #print taulist
            #print ccf

        else:
            taulist = np.asarray(taulist1)
            npts = np.asarray(npts1)
            ccf12 = np.asarray(ccf12)
            ccf21 = np.asarray(ccf21)
            ccf = (ccf12 + ccf21)*0.5

        '''
        try:
            ccf = (ccf12 + ccf21)*0.5

        except ValueError:

            f = open('error.txt', 'w')
            f.write(str(t1)+'\n')
            f.write(str(t2)+'\n')
            f.write(str(tau_temp1)+'\n')
            f.write(str(tau_temp2)+'\n')
            f.close()
        ''' 
    elif imode==1:
        taulist = np.asarray(taulist2)
        npts = asarray(npts2)
        ccf = ccf21 + 0.0
    else:
        taulist = np.asarray(taulist1)
        npts = asarray(npts1)
        ccf = ccf12 + 0.0
    
    return ccf, -taulist, npts




def peakcent(t1, y1, t2, y2, tlagmin, tlagmax, tunit, thres=0.8, siglevel=0.95, imode=0, sigmode = 0.2):
    '''
    Calculate peak time lag and centroid based on the cross-correlation 
    function for unevenly sampling data.
    
    Inputs:
        t1 -- time for light curve 1, assume increase;
        y1 -- flux for light curve 1;
        t2 -- time for light curve 2, assume increase;
        y2 -- flux for light curve 2;
        tlagmin -- minimum time lag;
        tlagmax -- maximum time lag;
        tunit -- tau step;
        thres -- lower limit of correlation coefficient when 
                 calculate centroid, default is 0.8;
        siglevel -- the required significant level of the 
                 correlation coefficient;
        imode -- cross-correlation mode: 0, twice (default); 
                 1, interpolate light curve 1;
                 2, interpolate light curve 2.
        sigmode -- how to deal with significance:
                Will use r = input value as the minimum correlation coefficient to consider (default = 0.2).
                0: Will use a p-test to assign significance to peak and discard peaks that are below
                the significance threshold (depends on number of points included and r). 
        
    Outputs:
        tlag_peak -- time lag based on the peak argument;
        status_peak -- peak status (1, constrained; 0, unconstrained);
        tlag_centroid -- time lag for centroid;
        status_centroid -- centroid status (1, constrained; 0, unconstrained);
    '''
    alpha = 1.0 - siglevel  # probability threshold to reject: no correlation hypothesis
    
    ccf_pack = xcor(t1, y1, t2, y2, tlagmin, tlagmax, tunit, imode)
    
    # ccf peaks --- excluding all with r < 0.2 instead of using p-value test. 
    if sigmode > 0:
        #print 'Using minimum r coefficient instead of significance test.'        
        max_indx = np.argmax(ccf_pack[0])
        max_rval = ccf_pack[0][max_indx]
        #Check and see if the max r is on the edge of the CCF. Fail it if so. 
            
        if np.logical_and(max_rval >= sigmode, np.logical_and(ccf_pack[1][max_indx] != tlagmin, ccf_pack[1][max_indx] != tlagmax)): 
            tlag_peak = ccf_pack[1][max_indx]
            max_rval = max_rval
            status_peak = 1
            status_rval = 1
        else:
            max_rval = -9999.0
            tlag_peak = -9999.0
            tlag_centroid = -9999.0
            status_peak = 0
            status_rval = 0
            status_centroid = 0

        #If the peak succeeds, calculate centroid:
        if status_peak == 1:
            rcent = thres*ccf_pack[0][max_indx]
            
            #populate arrays for centroid calculation!!
            lags_cent = []
            rval_cent = []
            temp_r = max_rval

            #start at r_max and lag_r_max
            temp_lag = ccf_pack[1][max_indx]
            temp_index = max_indx

            ## get centroid by 5pt method  ##jli184 nov.17.17
            
            if max_indx-2 < 0. or max_indx+3 > len(ccf_pack[1]):
                tlag_centroid = -9999.
                status_centroid = 0.
            else:
                lags_cent = ccf_pack[1][max_indx-2:max_indx+3]
                rval_cent = ccf_pack[0][max_indx-2:max_indx+3]
                tlag_centroid = np.sum(np.array(rval_cent)*np.array(lags_cent))/np.sum(np.array(rval_cent))
                status_centroid = 1
            
            ## get centroid by > rcent threshold          
            '''
            #Go higher in tau; if any of this fails, consider it a fail. 
            try:
                while temp_r >= rcent:
                    lags_cent.append(temp_lag)
                    rval_cent.append(temp_r) 

                    temp_index +=1
                    temp_r = ccf_pack[0][temp_index]
                    temp_lag = ccf_pack[1][temp_index]

                #Now start at the pixel to the left of the max
                temp_r = ccf_pack[0][max_indx-1]
                temp_lag = ccf_pack[1][max_indx-1]
                temp_index = max_indx-1
            
            #Now go downwards in lag
                while temp_r >= rcent:
                    lags_cent.append(temp_lag)
                    rval_cent.append(temp_r) 

                    temp_index -=1
                    temp_r = ccf_pack[0][temp_index]
                    temp_lag = ccf_pack[1][temp_index]

                lagr = zip(lags_cent, rval_cent)
                lagr.sort()
                lags_cent = [x for x, y in lagr]
                rval_cent = [x for y, x in lagr]
            
                tlag_centroid = np.sum(np.array(rval_cent)*np.array(lags_cent))/np.sum(np.array(rval_cent))
                status_centroid = 1                    
                
            except:
                tlag_centroid = -9999.0
                status_centroid = 0
                          
            #plt.plot(ccf_pack[1], ccf_pack[0], color = 'k')
            #plt.plot(lags_cent, rval_cent, color = 'r')
            #plt.axhline(y = rcent, color = 'r', linestyle = ':')
            #plt.axhline(y = max_rval, color = 'r', linestyle = '--')
            #plt.show()
            
            
            ## get centroid by > rcent threshold around local maximum ##jli184 nov/17/17            
            #Go higher in tau; if any of this fails, consider it a fail. 
            try:
                for i in range(len(ccf_pack[0])):
                    if ccf_pack[0][max_indx+i] >= rcent:
                        lags_cent.append(ccf_pack[1][max_indx+i])
                        rval_cent.append(ccf_pack[0][max_indx+i]) 
                    else:
                        break;

                for i in range(1,len(ccf_pack[0])):
                    if ccf_pack[0][max_indx-i] >= rcent:
                        lags_cent.append(ccf_pack[1][max_indx-i])
                        rval_cent.append(ccf_pack[0][max_indx-i]) 
                    else:
                        break;


                lagr = zip(lags_cent, rval_cent)
                lagr.sort()
                lags_cent = [x for x, y in lagr]
                rval_cent = [x for y, x in lagr]
            
                tlag_centroid = np.sum(np.array(rval_cent)*np.array(lags_cent))/np.sum(np.array(rval_cent))
                status_centroid = 1                    
                
            except:
                tlag_centroid = -9999.0
                status_centroid = 0
                          
            plt.plot(ccf_pack[1], ccf_pack[0], color = 'k')
            plt.plot(lags_cent, rval_cent, 'r.')
            plt.axhline(y = rcent, color = 'r', linestyle = ':')
            plt.axhline(y = max_rval, color = 'r', linestyle = '--')
            plt.show()
            '''
                      
        if ccf_pack[2][max_indx]>2.0:
            peak_pvalue = corsig(ccf_pack[0][max_indx], float(ccf_pack[2][max_indx]-2.0))
        else:
            peak_pvalue = 1.0
        
        #Now, if the centroid fails, re-set the peak status to 0 because we don't want to report a peak without a centroid!
        if status_centroid == 0:
            status_peak = 2
            tlag_peak = -999.0
            max_rval = -9999.0
            status_rval = 0 

    # ccf peaks-- Eric's method using a p-value test (usually not using) 
    else:
        max_indx = np.argmax(ccf_pack[0])
        if ccf_pack[2][max_indx]>2.0:
            peak_pvalue = corsig(ccf_pack[0][max_indx], float(ccf_pack[2][max_indx]-2.0))
        else:
            peak_pvalue = 1.0

        if peak_pvalue<alpha:
            max_rval = ccf_pack[0][max_indx]
            tlag_peak = ccf_pack[1][max_indx]
            status_peak = 1
            status_rval = 1
        else:
            max_rval = -9999.0
            tlag_peak = -9999.0
            status_peak = 0
            status_rval = 0
            tlag_centroid = -9999.0
            status_centroid = 0
    
        if status_peak == 1:
            rcent = thres*ccf_pack[0][max_indx]
            selcen = ccf_pack[0]>rcent # points to calculate centroid
            if np.sum(selcen)>1 and ccf_pack[0][0]<rcent and ccf_pack[0][-1]<rcent and status_peak==1:
                tlag_centroid = np.sum(ccf_pack[0][selcen]*ccf_pack[1][selcen])/np.sum(ccf_pack[0][selcen])
                status_centroid = 1
            else:
                tlag_centroid = -9999.0
                status_centroid = 0

    #print tlag_peak, status_peak, tlag_centroid, status_centroid, max_rval, status_rval
    return tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, peak_pvalue




def xcor_mc(t1, y1, dy1, t2, y2, dy2, tlagmin, tlagmax, tunit, thres=0.8, siglevel=0.95, imode=0, nsim=2048, mcmode=0, sigmode = 0.2):
    '''
    Calculate the uncertainty for the cross-correlation peak.
    
    Inputs:
        t1 -- time for light curve 1, assume increase;
        y1 -- flux for light curve 1;
        dy1 -- flux uncertainty for light curve 1;
        t2 -- time for light curve 2, assume increase;
        y2 -- flux for light curve 2;
        dy2 -- flux uncertainty for light curve 2;
        tlagmin -- minimum time lag;
        tlagmax -- maximum time lag;
        tunit -- tau step;
        thres -- lower limit of correlation coefficient when 
                 calculate centroid, default is 0.8;
        siglevel -- the required significant level of the 
                 correlation coefficient;
        imode -- cross-correlation mode: 0, twice (default); 
                 1, interpolate light curve 1;
                 2, interpolate light curve 2.
        nsim -- MC simulation trials;
        mcmode -- MC mode: 0, RSS plus FR
                  1, RSS only
                  2, FR only
        sigmode -- How to exclude non-significant peaks:
                  Will exclude all peaks with r < input value 
                  0 will exclude all peaks based on p-value significance test. 
        
    Outputs:
        tlags_peak -- tlag of peak distribution;
        tlags_centroid -- tlag of centroid distribution;
        nsuccess_peak -- success times in finding peaks;
        nfail_peak -- fail times in finding peaks;
        nsuccess_centroid -- success times in calculating centroid;
        nfail_centroid -- fail times in calculating centroid.
    '''
    numt1 = len(t1)
    numt2 = len(t2)
    tlags_peak = []
    tlags_centroid = []
    pvals = []
    nsuccess_peak = 0
    nsuccess_rvals = 0
    nfail_peak = 0
    nsuccess_centroid = 0
    nfail_centroid = 0
    nfail_rvals = 0
    max_rvals = []
    for i in xrange(nsim):
        if mcmode!=2:
            # RSS resample light curve 1
            indx1 = np.random.randint(0, numt1, numt1)
            unique1, counts1 = np.unique(indx1, return_counts=True) # sorted unique value
            t1_rss = t1[unique1]
            y1_rss = y1[unique1]
            dy1_rss = dy1[unique1]/np.sqrt(counts1)
            
            indx2 = np.random.randint(0, numt2, numt2)
            unique2, counts2 = np.unique(indx2, return_counts=True)
            t2_rss = t2[unique2]
            y2_rss = y2[unique2]
            dy2_rss = dy2[unique2]/np.sqrt(counts2)
            
        else:
            # do not apply RSS resample, rss light curve equals to original one
            t1_rss = t1 + 0.0
            y1_rss = y1 + 0.0
            dy1_rss = dy1 + 0.0
            
            t2_rss = t2 + 0.0
            y2_rss = y2 + 0.0
            dy2_rss = dy2 + 0.0
            
        
        if mcmode!=1:
            # measurement error perturbation
            t1_fr = t1_rss + 0.0
            y1_fr = np.random.normal(y1_rss, dy1_rss)
            t2_fr = t2_rss + 0.0
            y2_fr = np.random.normal(y2_rss, dy2_rss)
        else:
            # do not aplly the error perturbation
            t1_fr = t1_rss + 0.0
            y1_fr = y1_rss + 0.0
            t2_fr = t2_rss + 0.0
            y2_fr = y2_rss + 0.0
        
        # perform CCF
        pc_pack = peakcent(t1_fr, y1_fr, t2_fr, y2_fr, tlagmin, tlagmax, tunit, thres, siglevel, imode, sigmode = sigmode)
        
        # ccf peaks
        if pc_pack[1]==1:
            tau_peak = pc_pack[0]
            tlags_peak.append(tau_peak)
            pval = pc_pack[7]
            pvals.append(pval)
            nsuccess_peak += 1
        elif pc_pack[1] == 0:
            nfail_peak += 1
        
        # ccf centroids
        if pc_pack[3]==1:
            tau_centroid = pc_pack[2]
            tlags_centroid.append(tau_centroid)
            nsuccess_centroid += 1
        else:
            nfail_centroid += 1
            
        # max_rvalues
        if pc_pack[6]==1:
            max_rvals.append(pc_pack[5])
            nsuccess_rvals += 1
        else:
            nfail_rvals += 1

    
    tlags_peak = np.asarray(tlags_peak)
    tlags_centroid = np.asarray(tlags_centroid)
    print ('Failed centroids: ', nfail_centroid)
    print ('Failed peaks: ', nfail_peak)
    
    return tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals

