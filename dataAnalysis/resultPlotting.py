'''
Created on 24 Sep 2014

@author: mkaminskas
'''
'''
plot the surprise values against profile size for individual users from all iterations
also fits the least squares polynomial curve

so the method has some repetitions. who cares, it's only 4 plots
'''
# import scipy.stats as stat

import ast
from itertools import izip
import itertools
import operator
import os
import pickle

from statsmodels.stats.weightstats import ttest_ind

from dataHandling import dataReading, dataPreprocessing
from dataModel import trainData
from frameworkMetrics import novelty, explanationMetrics
import matplotlib.pyplot as plt
import numpy as np
from utils import config, counter


def plotProfileSize():
    '''
    plotting the relation between surprise values and user's profile size
    '''
     
    profile_size = []
    s_cont = []
    s_rat = []
     
    profile_size_avg = []
    s_cont_avg = []
    s_rat_avg = []
    
    iterations = 5
     
    for i in range(1,iterations+1):
        with open('../rez/recsys paper/LFM w profile/raw_lastfm_factor_baseline_10_5_'+str(i)+'.dat','rb') as f:
            for line in f:
                dataReading = line.split(':')
                # skip outliers
                if float(dataReading[1]) < 1000:
                    profile_size.append(float(dataReading[1]))
                    s_cont.append(float(dataReading[3]))
                    s_rat.append(float(dataReading[4]))
        with open('../rez/recsys paper/LFM avg w profile/AVG_raw_lastfm_factor_baseline_10_5_'+str(i)+'.dat','rb') as f_avg:
            for line in f_avg:
                dataReading = line.split(':')
                if float(dataReading[1]) < 1000:
                    profile_size_avg.append(float(dataReading[1]))
                    s_cont_avg.append(float(dataReading[3]))
                    s_rat_avg.append(float(dataReading[4]))
     
    x_data = np.array(profile_size)
    y_data_cont = np.array(s_cont)
    y_data_rat = np.array(s_rat)
     
    coefficients_c = np.polyfit(x_data, y_data_cont, 1)
    polynomial_c = np.poly1d(coefficients_c)
    coefficients_r = np.polyfit(x_data, y_data_rat, 1)
    polynomial_r = np.poly1d(coefficients_r)
    x_fit = np.arange(0, 1000, 10)
    y_fit_c = polynomial_c(x_fit)
    y_fit_r = polynomial_r(x_fit)
     
    plt.subplot(221)
    plt.plot(x_fit, y_fit_c,'#ffbfbf', linewidth=2.0)
    plt.plot(x_fit, y_fit_r,'#ffbfbf', linewidth=2.0)
    plt.scatter(profile_size, s_cont, s=20, c='b', label="$S_{cont}$")
    plt.scatter(profile_size, s_rat, s=20, c='w', label="$S_{co-occ}$")
    plt.legend(loc=(0.7,0.7), prop={'size':22})
#     plt.xlabel('Profile size')
    plt.xlim(0.0, 1000)
    plt.xticks(np.arange(0, 1001, 100))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Lower-bound distance surprise values. LastFM', fontsize=20)
     
     
    x_data_avg = np.array(profile_size_avg)
    y_data_cont_avg = np.array(s_cont_avg)
    y_data_rat_avg = np.array(s_rat_avg)
     
    coefficients_c = np.polyfit(x_data_avg, y_data_cont_avg, 1)
    polynomial_c = np.poly1d(coefficients_c)
    coefficients_r = np.polyfit(x_data_avg, y_data_rat_avg, 1)
    polynomial_r = np.poly1d(coefficients_r)
    x_fit = np.arange(0, 1000, 10)
    y_fit_c = polynomial_c(x_fit)
    y_fit_r = polynomial_r(x_fit)
     
    plt.subplot(222)
    plt.plot(x_fit, y_fit_c,'#ffbfbf', linewidth=2.0)
    plt.plot(x_fit, y_fit_r,'#ffbfbf', linewidth=2.0)
    plt.scatter(profile_size_avg, s_cont_avg, s=20, c='b', label="$S_{cont}^{avg}$")
    plt.scatter(profile_size_avg, s_rat_avg, s=20, c='w', label="$S_{co-occ}^{avg}$")
    plt.legend(loc=(0.7,0.5), prop={'size':22})
#     plt.xlabel('Profile size')
    plt.xlim(0.0, 1000)
    plt.xticks(np.arange(0, 1001, 100))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Average distance surprise values. LastFM', fontsize=20)
     
    ################## Same procedure for MovieLens ###########################
     
    profile_size2 = []
    s_cont2 = []
    s_rat2 = []
     
    profile_size_avg2 = []
    s_cont_avg2 = []
    s_rat_avg2 = []
     
    for i in range(1,iterations+1):
        with open('../rez/recsys paper/MOV w profile/raw_movielens_factor_baseline_10_5_'+str(i)+'.dat','rb') as f:
            for line in f:
                dataReading = line.split(':')
                if float(dataReading[1]) < 400:
                    profile_size2.append(float(dataReading[1]))
                    s_cont2.append(float(dataReading[3]))
                    s_rat2.append(float(dataReading[4]))
        with open('../rez/recsys paper/MOV avg w profile/AVG_raw_movielens_factor_baseline_10_5_'+str(i)+'.dat','rb') as f_avg:
            for line in f_avg:
                dataReading = line.split(':')
                if float(dataReading[1]) < 400:
                    profile_size_avg2.append(float(dataReading[1]))
                    s_cont_avg2.append(float(dataReading[3]))
                    s_rat_avg2.append(float(dataReading[4]))
     
    x_data = np.array(profile_size2)
    y_data_cont = np.array(s_cont2)
    y_data_rat = np.array(s_rat2)
     
    coefficients_c = np.polyfit(x_data, y_data_cont, 1)
    polynomial_c = np.poly1d(coefficients_c)
    coefficients_r = np.polyfit(x_data, y_data_rat, 1)
    polynomial_r = np.poly1d(coefficients_r)
    x_fit = np.arange(0, 410, 10)
    y_fit_c = polynomial_c(x_fit)
    y_fit_r = polynomial_r(x_fit)
     
    plt.subplot(223)
    plt.plot(x_fit, y_fit_c,'#ffbfbf', linewidth=2.0)
    plt.plot(x_fit, y_fit_r,'#ffbfbf', linewidth=2.0)
    plt.scatter(profile_size2, s_cont2, s=20, c='b', label="$S_{cont}$")
    plt.scatter(profile_size2, s_rat2, s=20, c='w', label="$S_{co-occ}$")
    plt.legend(loc=(0.7,0.05), prop={'size':22})
    plt.xlabel('Profile size', fontsize=15)
    plt.xlim(0, 400)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Lower-bound distance surprise values. MovieLens', fontsize=20)
     
    x_data_avg = np.array(profile_size_avg2)
    y_data_cont_avg = np.array(s_cont_avg2)
    y_data_rat_avg = np.array(s_rat_avg2)
     
    coefficients_c = np.polyfit(x_data_avg, y_data_cont_avg, 1)
    polynomial_c = np.poly1d(coefficients_c)
    coefficients_r = np.polyfit(x_data_avg, y_data_rat_avg, 1)
    polynomial_r = np.poly1d(coefficients_r)
    x_fit = np.arange(0, 410, 10)
    y_fit_c = polynomial_c(x_fit)
    y_fit_r = polynomial_r(x_fit)
     
    plt.subplot(224)
    plt.plot(x_fit, y_fit_c,'#ffbfbf', linewidth=2.0)
    plt.plot(x_fit, y_fit_r,'#ffbfbf', linewidth=2.0)
    plt.scatter(profile_size_avg2, s_cont_avg2, s=20, c='b', label="$S_{cont}^{avg}$")
    plt.scatter(profile_size_avg2, s_rat_avg2, s=20, c='w', label="$S_{co-occ}^{avg}$")
    plt.legend(loc=(0.7,0.5), prop={'size':22})
    plt.xlabel('Profile size', fontsize=15)
    plt.xlim(0, 400)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Average distance surprise values. MovieLens', fontsize=20)
    
    plt.show()


def plotBaselineChart_SelectedMethods(path, iterations, plot_coverage=False):
    '''
    plotting the baseline (no re-ranking) results for 3 selected algorithms
    '''
    
    if 'movies' in path:
        algorithms = ['warp_25_1000.0', 'mf_25', 'ibnon_normalized_60', 'ubnon_normalized_150']
    elif 'music' in path:
        algorithms = ['warp_50_10.0', 'mf_25', 'ibnon_normalized_60', 'ubnon_normalized_20']
    
    full_data = []
    
    for algorithm in algorithms:
        
        coverage_path = None
        if plot_coverage:
            if 'movies' in path:
                coverage_path = path.replace('_movies','_movies/coverage') + 'movies_' + algorithm + '_baseline'
            elif 'music' in path:
                coverage_path = path.replace('_music','_music/coverage') + 'music_' + algorithm + '_baseline'
        
        if 'movies' in path:
            data = computeDataStatistics(path+'movies_'+algorithm+'_baseline_', iterations, coverage_path)
        elif 'music' in path:
            data = computeDataStatistics(path+'music_'+algorithm+'_baseline_', iterations, coverage_path)
        
        full_data.append(data)
    
    width = 0.15
    fig, ax = plt.subplots()
    
    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx(), ax.twinx()]
    
    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.75)
    
    # Move the last y-axis spine over to the right by 20% of the width of the axes
    axes[-1].spines['right'].set_position(('axes', 1.1))
    
    # To make the border of the right-most axis visible, we need to turn the frame on.
    # This hides the other plots, however, so we need to turn its fill off.
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    
    warp_color = '0.0'
    mf_color = '0.25'
    ib_color = '0.5'
    ub_color = '0.75'
    
    # the order of frameworkMetrics: 1) Recall, 2) Div_cont, 3) Div_r, 4) Sur_cont, 5) Sur_r, 6) Novelty, 7) Coverage
    label_list = ('$Recall$', '$Div_{cont}$', '$Div_{ratings}$','$S_{cont}$', '$S_{co-occ}$', '$Novelty$', '$Coverage$')
    if 'movies' in path:
        metric_list_1 = ['recall', 'diversity_r', 'surprise_r', 'novelty']
        tick_list_1 = [1, 3, 5, 6]
        metric_list_2 = ['diversity_c', 'surprise_c']
        tick_list_2 = [2, 4]
    elif 'music' in path:
        metric_list_1 = ['recall', 'diversity_r', 'surprise_r',  'surprise_c', 'novelty']
        tick_list_1 = [1, 3, 5, 4, 6]
        metric_list_2 = ['diversity_c']
        tick_list_2 = [2]
    
    means = [full_data[0][metric][0] for metric in metric_list_1]
    axes[0].bar(np.array(tick_list_1)+(width), means, width, color=warp_color)
    
    means = [full_data[1][metric][0] for metric in metric_list_1]
    axes[0].bar(np.array(tick_list_1)+(width*2), means, width, color=mf_color)
    
    # check significance against first bar
    for metric in metric_list_1:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[1][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. mf',p
    
    
    means = [full_data[2][metric][0] for metric in metric_list_1]
    axes[0].bar(np.array(tick_list_1)+(width*3), means, width, color=ib_color)
    
    # check significance against first two bars
    for metric in metric_list_1:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[2][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. ib',p
        
        _, p, _ = ttest_ind(full_data[1][metric+'_full'], full_data[2][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'mf vs. ib',p
    
    means = [full_data[3][metric][0] for metric in metric_list_1]
    axes[0].bar(np.array(tick_list_1)+(width*4), means, width, color=ub_color)
    
    # check significance against first three bars
    for metric in metric_list_1:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. ub',p
        
        _, p, _ = ttest_ind(full_data[1][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'mf vs. ub',p
        
        _, p, _ = ttest_ind(full_data[2][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'ib vs. ub',p
    
    axes[0].set_ylim(0.0, 0.5)
    
    #######
    
    means = [full_data[0][metric][0] for metric in metric_list_2]
    axes[1].bar(np.array(tick_list_2)+(width), means, width, color=warp_color)
    
    means = [full_data[1][metric][0] for metric in metric_list_2]
    axes[1].bar(np.array(tick_list_2)+(width*2), means, width, color=mf_color)
    
    # check significance against first bar
    for metric in metric_list_2:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[1][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. mf',p
    
    
    means = [full_data[2][metric][0] for metric in metric_list_2]
    axes[1].bar(np.array(tick_list_2)+(width*3), means, width, color=ib_color)
    
    # check significance against first two bars
    for metric in metric_list_2:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[2][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. ib',p
        
        _, p, _ = ttest_ind(full_data[1][metric+'_full'], full_data[2][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'mf vs. ib',p
    
    means = [full_data[3][metric][0] for metric in metric_list_2]
    axes[1].bar(np.array(tick_list_2)+(width*4), means, width, color=ub_color)
    
    # check significance against first three bars
    for metric in metric_list_2:
        _, p, _ = ttest_ind(full_data[0][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'warp vs. ub',p
        
        _, p, _ = ttest_ind(full_data[1][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'mf vs. ub',p
        
        _, p, _ = ttest_ind(full_data[2][metric+'_full'], full_data[3][metric+'_full'])
        if p < 0.01:
            print 'Difference is significant',metric,'ib vs. ub',p
    
    if 'movies' in path:
        axes[1].set_ylim(0.85, 0.98)
    elif 'music' in path:
        axes[1].set_ylim(0.8, 0.85)
    
    #######
    
    means = [full_data[0]['coverage'][0]]
    warp_data = axes[2].bar(np.array([7])+(width), means, width, color=warp_color)
    
    means = [full_data[1]['coverage'][0]]
    mf_data = axes[2].bar(np.array([7])+(width*2), means, width, color=mf_color)
    
    means = [full_data[2]['coverage'][0]]
    ib_data = axes[2].bar(np.array([7])+(width*3), means, width, color=ib_color)
    
    means = [full_data[3]['coverage'][0]]
    ub_data = axes[2].bar(np.array([7])+(width*4), means, width, color=ub_color)
    
    
    # add some text for labels, title and axes ticks
    if 'movies' in path:
        axes[0].set_ylabel('$Recall \quad Div_{ratings} \quad S_{co-occ} \quad Novelty$', fontsize=20)
        axes[1].set_ylabel('$Div_{cont} \quad S_{cont}$', fontsize=20)
    elif 'music' in path:
        axes[0].set_ylabel('$Recall \quad Div_{ratings} \quad S_{co-occ} \quad S_{cont} \quad Novelty$', fontsize=20)
        axes[1].set_ylabel('$Div_{cont}$', fontsize=20)
    
    axes[2].set_ylabel('$Coverage$', fontsize=20)
    ax.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7])+4*width)
    ax.set_xticklabels( label_list, fontsize=17 )
    
    ax.legend( (warp_data[0], mf_data[0], ib_data[0], ub_data[0], ), ('LTR', 'MF', 'IB', 'UB'), loc=3, prop={'size':17} )
    
    plt.show()



def plotBaselineChart(path, iterations, plot_coverage=False):
    '''
    plotting the baseline (no re-ranking) results for all implemented algorithms
    '''
    
#     algorithms = ['ubnon_normalized_20', 'ubnon_normalized_30', 'ubnon_normalized_40', 'ubnon_normalized_50', 'ubnon_normalized_60', 'ubnon_normalized_70', \
#                   'ubnon_normalized_80', 'ubnon_normalized_90', 'ubnon_normalized_100', 'ubnon_normalized_150', 'ubnon_normalized_175', 'ubnon_normalized_200', \
#                   'ubnon_normalized_250']
#     algorithms = ['warp_25_0.1', 'warp_25_0.01', 'warp_25_0.001', 'warp_50_0.1', 'warp_50_0.01', 'warp_50_0.001', 'warp_75_0.1', 'warp_75_0.01', 'warp_75_0.001', \
#                         'warp_100_0.1', 'warp_100_0.01', 'warp_100_0.001',]
#     algorithms = ['warp_25_1000.0', 'warp_50_10.0', 'warp_100_1.0', 'warp_100_100.0', 'warp_50_10.2']
#     algorithms = ['ibnon_normalized_20', 'ibnon_normalized_30', 'ibnon_normalized_40', 'ibnon_normalized_50', 'ibnon_normalized_60', 'ibnon_normalized_70', \
#                   'ibnon_normalized_80', 'ibnon_normalized_90', 'ibnon_normalized_100', 'ibnon_normalized_150', 'ibnon_normalized_175', 'ibnon_normalized_200', \
#                   'ibnon_normalized_250']
    algorithms = ['mf_25', 'mf_50', 'mf_75', 'mf_100', 'mf_150', 'mf_175', 'mf_200', 'mf_250']
    
    full_data = []
    
    for algorithm in algorithms:
        
        coverage_path = None
        if plot_coverage:
            if 'movies' in path:
                coverage_path = path.replace('_movies','_movies/coverage') + 'movies_' + algorithm + '_baseline'
            elif 'music' in path:
                coverage_path = path.replace('_music','_music/coverage') + 'music_' + algorithm + '_baseline'
        
        if 'movies' in path:
            data = computeDataStatistics(path+'movies_'+algorithm+'_baseline_', iterations, coverage_path)
        elif 'music' in path:
            data = computeDataStatistics(path+'music_'+algorithm+'_baseline_', iterations, coverage_path)
        
        full_data.append(data)
    
    
    width = 0.07
    fig, ax = plt.subplots()
    
    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx(), ax.twinx()]
    
    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.75)
    
    # Move the last y-axis spine over to the right by 20% of the width of the axes
    axes[-1].spines['right'].set_position(('axes', 1.1))
    
    # To make the border of the right-most axis visible, we need to turn the frame on.
    # This hides the other plots, however, so we need to turn its fill off.
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)
    
    color_list = np.arange(0,1,0.1)
    hatch_list = ['//', '.', 'x', 'o']
    
    # the order of frameworkMetrics: 1) Recall, 2) Div_cont, 3) Div_r, 4) Sur_cont, 5) Sur_r, 6) Novelty, 7) Coverage
    label_list = ('$Recall$', '$Div_{cont}$', '$Div_{ratings}$','$S_{cont}$', '$S_{co-occ}$', '$Novelty$', '$Coverage$')
    if 'movies' in path:
        metric_list_1 = ['recall', 'diversity_r', 'surprise_r', 'novelty']
        tick_list_1 = [1, 3, 5, 6]
        metric_list_2 = ['diversity_c', 'surprise_c']
        tick_list_2 = [2, 4]
    elif 'music' in path:
        metric_list_1 = ['recall', 'diversity_r', 'surprise_r',  'surprise_c', 'novelty']
        tick_list_1 = [1, 3, 5, 4, 6]
        metric_list_2 = ['diversity_c']
        tick_list_2 = [2]
    
    
    for counter in range(1,len(full_data)+1):
        means = [full_data[counter-1][metric][0] for metric in metric_list_1]
        if counter >= len(color_list):
            # if the number of methods is more than the number of colors, use the last color and hatch
            axes[0].bar(np.array(tick_list_1)+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)])
        else:
            # else, just use the color
            axes[0].bar(np.array(tick_list_1)+width*counter, means, width, color=str(color_list[counter-1]))
    
    axes[0].set_ylim(0.0, 0.5)
    
    for counter in range(1,len(full_data)+1):
        means = [full_data[counter-1][metric][0] for metric in metric_list_2]
        if counter >= len(color_list):
            # if the number of methods is more than the number of colors, use the last color and hatch
            axes[1].bar(np.array(tick_list_2)+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)])
        else:
            # else, just use the color
            axes[1].bar(np.array(tick_list_2)+width*counter, means, width, color=str(color_list[counter-1]))
    
    if 'movies' in path:
        axes[1].set_ylim(0.85, 0.98)
    elif 'music' in path:
        axes[1].set_ylim(0.8, 0.85)
    
    labels = []
    for counter in range(1,len(full_data)+1):
        means = [full_data[counter-1]['coverage'][0]]
        if counter >= len(color_list):
            # if the number of methods is more than the number of colors, use the last color and hatch
            labels.append( axes[2].bar(np.array([7])+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)]) )
        else:
            # else, just use the color
            labels.append( axes[2].bar(np.array([7])+width*counter, means, width, color=str(color_list[counter-1])) )
    
    # add some text for labels, title and axes ticks
    if 'movies' in path:
        axes[0].set_ylabel('$Recall \quad Div_{ratings} \quad S_{co-occ} \quad Novelty$', fontsize=20)
        axes[1].set_ylabel('$Div_{cont} \quad S_{cont}$', fontsize=20)
    elif 'music' in path:
        axes[0].set_ylabel('$Recall \quad Div_{ratings} \quad S_{co-occ} \quad S_{cont} \quad Novelty$', fontsize=20)
        axes[1].set_ylabel('$Div_{cont}$', fontsize=20)
    
    axes[2].set_ylabel('$Coverage$', fontsize=20)
    ax.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7])+4*width)
    ax.set_xticklabels( label_list, fontsize=17 )
    
#     ax.legend( (warp_data[0], mrec_data[0], mf_data[0], ib_data[0], ub_data[0], ub_data2[0], ub_data3[0], ub_data4[0], ub_data5[0], ub_data6[0], ub_data7[0], \
#                 ub_data8[0]), ('25_0.1', '25_0.01','25_0.001', '50_0.1', '50_0.01', '50_0.001', '75_0.1', '75_0.01', '75_0.001', '100_0.1', '100_0.01', \
#                                '100_0.001'), loc=2, prop={'size':16} )
    
    
    ax.legend( [label[0] for label in labels], ['k='+alg_name.split('_')[-1] for alg_name in algorithms], loc=3, prop={'size':14} )
    
    plt.show()



def plotFullChart(path, iterations, plot_coverage=False, feking_surprise_iterations=5):
    '''
    plotting the re-ranking results for a given algorithm
    '''
    
    full_data = []
    
    # for each re-ranking method, read the metric values
    for method in ['baseline_', 'diversity-content_0.5_', 'diversity-ratings_0.5_', 'surprise-content_0.5_', 'surprise-ratings_0.5_', 'novelty_0.5_']:
#     for method in ['baseline_', 'diversity-content_0.5_', 'diversity-ratings_0.5_', 'surprise-content_0.5_', 'surprise-ratings_0.5_', 'novelty_0.5_', \
#                    'expl_accuracy_', 'expl_coverage_', 'expl_lift_']:
        
        coverage_path = None
        if plot_coverage:
            if 'movies' in path:
                coverage_path = path.replace('_movies','_movies/coverage') + '_' + method.rstrip('_0.5_')
            elif 'music' in path:
                coverage_path = path.replace('_music','_music/coverage') + '_' + method.rstrip('_0.5_')
        
        # a workaround for surprise lagging on itertions:
        if 'surprise' in method:
            stats = computeDataStatistics(path+'_'+method, feking_surprise_iterations, coverage_path)
        else:
            stats = computeDataStatistics(path+'_'+method, iterations, coverage_path, verbose=True)
        
        full_data.append(stats)
    
    if 'mf' in path:
        color = '0.25'
    elif 'ib' in path:
        color = '0.5'
    elif 'ub' in path:
        color = '0.75'
    elif 'warp' in path:
        color = '0.0'
    elif 'mrec' in path:
        color = '1.0'
    
    width = 0.45
    x_labels = ('$Baseline^r$', '$Div_{cont}^r$', '$Div_{ratings}^r$', '$S_{cont}^r$', '$S_{co-occ}^r$', '$Novelty^r$', '$Expl_{acc}^r$', '$Expl_{cov}^r$', '$Expl_{lift}^r$')
    indices = np.arange(1, len(full_data)+1)
    errorbar_style = dict(elinewidth=2,ecolor='black',capsize=4,capthick=2)
    
    means = [reranking_data['recall'][0] for reranking_data in full_data]
    plt.subplot(431)
    plt.bar(indices, means, width=width, color=color)
    plt.ylabel('a) $Recall$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
#     plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.ylim(0.25, 0.45) # that's for latest experiments
#     plt.ylim(0.15, 0.40)
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    means = [reranking_data['diversity_c'][0] for reranking_data in full_data]
    devs = [reranking_data['diversity_c'][1] for reranking_data in full_data]
    plt.subplot(432)
    plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
    plt.ylabel('b) $Div_{cont}$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
    plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    means = [reranking_data['diversity_r'][0] for reranking_data in full_data]
    devs = [reranking_data['diversity_r'][1] for reranking_data in full_data]
    plt.subplot(433)
    plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
    plt.ylabel('c) $Div_{ratings}$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
    plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    means = [reranking_data['surprise_c'][0] for reranking_data in full_data]
    devs = [reranking_data['surprise_c'][1] for reranking_data in full_data]
    plt.subplot(434)
    plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
    plt.ylabel('d) $S_{cont}$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
    plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    means = [reranking_data['surprise_r'][0] for reranking_data in full_data]
    devs = [reranking_data['surprise_r'][1] for reranking_data in full_data]
    plt.subplot(435)
    plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
    plt.ylabel('e) $S_{co-occ}$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
    plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    means = [reranking_data['novelty'][0] for reranking_data in full_data]
    devs = [reranking_data['novelty'][1] for reranking_data in full_data]
    plt.subplot(436)
    plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
    plt.ylabel('f) $Novelty$', fontsize=17)
    plt.xticks(indices+width/2., x_labels, fontsize=15)
    plt.xlim(0.5, len(full_data) + 1.0)
    plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
    plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    if plot_coverage:
        means = [reranking_data['coverage'][0] for reranking_data in full_data]
        plt.subplot(438)
        plt.bar(indices, means, width=width, color=color, yerr=devs, error_kw=errorbar_style)
        plt.ylabel('g) $Coverage$', fontsize=17)
        plt.xticks(indices+width/2., x_labels, fontsize=15)
        plt.xlim(0.5, len(full_data) + 1.0)
#         plt.ylim(round(max(0.0, 1.5*min(means)-0.5*max(means)),2), round(1.5*max(means) - 0.5*min(means),2))
        plt.ylim(300, 1800)  # that's for latest experiments
#         plt.ylim(100, 900)
        plt.grid(b=True, axis='y', which='major', color='b', linestyle='--')
    
    plt.show()



def computeDataStatistics(path, iterations, coverage_file=None, verbose=False):
    '''
    get the mean and standard deviation of the different frameworkMetrics for a given algorithm:
    for each metric store (mean, variance) tuples
    '''
    
    recall = []
    diversity_c = []
    diversity_r = []
    surprise_c = []
    surprise_r = []
#     surprise_r_n = []
    novelty = []
    overlap = []
    
    means_and_stds = {}
    
    for i in range(1,iterations+1):
        filepath = path+str(i)+'.dat'
        with open(filepath,'rb') as data_file:
            for line in data_file:
                dataReading = line.split(':')
                recall.append(float(dataReading[2]))
                diversity_c.append(float(dataReading[3]))
                diversity_r.append(float(dataReading[4]))
                surprise_c.append(float(dataReading[5]))
                surprise_r.append(float(dataReading[6]))
#                 surprise_r_n.append(float(dataReading[7]))
                novelty.append(float(dataReading[8]))
                overlap.append(float(dataReading[9]))
                
                # needed for t- or z-test
                means_and_stds.setdefault('recall_full',[]).append(float(dataReading[2]))
                means_and_stds.setdefault('diversity_c_full',[]).append(float(dataReading[3]))
                means_and_stds.setdefault('diversity_r_full',[]).append(float(dataReading[4]))
                means_and_stds.setdefault('surprise_c_full',[]).append(float(dataReading[5]))
                means_and_stds.setdefault('surprise_r_full',[]).append(float(dataReading[6]))
                means_and_stds.setdefault('novelty_full',[]).append(float(dataReading[8]))
    
    means_and_stds['recall'] = (np.mean(recall), np.std(recall))
    means_and_stds['diversity_c'] = (np.mean(diversity_c), np.std(diversity_c))
    means_and_stds['diversity_r'] = (np.mean(diversity_r), np.std(diversity_r))
    means_and_stds['surprise_c'] = (np.mean(surprise_c), np.std(surprise_c))
    means_and_stds['surprise_r'] = (np.mean(surprise_r), np.std(surprise_r))
#     means_and_stds['surprise_r_n'] = (np.mean(surprise_r_n), np.std(surprise_r_n))
    means_and_stds['novelty'] = (np.mean(novelty), np.std(novelty))
    means_and_stds['overlap'] = (np.mean(overlap), np.std(overlap))
    
    coverage = []
    if coverage_file:
        with open(coverage_file+'.dat','rb') as cov_file:
            for line in cov_file:
                if (len(line) > 2) and ('iteration' not in line):
                    coverage.append(float(line))
            means_and_stds['coverage'] = ( np.mean(coverage), np.std(coverage) )
    
    if verbose:
        print '----------',path,'----------\n'
        print 'recall',means_and_stds['recall']
        print 'diversity_c',means_and_stds['diversity_c']
        print 'diversity_r',means_and_stds['diversity_r']
        print 'surprise_c',means_and_stds['surprise_c']
        print 'surprise_r',means_and_stds['surprise_r']
#         print 'surprise_r_n',means_and_stds['surprise_r_n']
        print 'novelty',means_and_stds['novelty']
        print 'overlap',means_and_stds['overlap']
        print '\n'
    
    return means_and_stds


def checkTags(movies_or_music, threshold):
    '''
    print the details on item content labels 
    '''
    
    config.MOVIES_OR_MUSIC = movies_or_music
    dataReading._readItemData()
    
#     tag_frequencies = counter.Counter([label.lower() for labels in config.ITEM_DATA.values() for label in labels])
    tag_frequencies = counter.Counter([label.lower() for item_dict in config.ITEM_DATA.values() for label in item_dict['labels']])
    frequent_tags = [label for label, freq in tag_frequencies.items() if freq >= threshold]
    
    print 'f >=',threshold
    print 'tags before',len(tag_frequencies), 'tags after',len(frequent_tags)
    
    new_movie_labels = {}
    
    for movie, movie_data in config.ITEM_DATA.iteritems():
        new_movie_labels[movie] = [l for l in movie_data['labels'] if l.lower() in frequent_tags]
    
    print len(config.ITEM_DATA), len(new_movie_labels)
    
    movie_coverage = [len(l) for dict['labels'] in config.ITEM_DATA.values()]
    new_movie_coverage = [len(l) for l in new_movie_labels.values()]
    
    print 'mean/min/max num. of labels BEFORE FREQUENCY FILTERING', np.mean(movie_coverage), min(movie_coverage), max(movie_coverage)
    print 'mean/min/max num. of labels AFTER FREQUENCY FILTERING', np.mean(new_movie_coverage), min(new_movie_coverage), max(new_movie_coverage)
    
    for k in range(2):
        print '----------------'
        print 'movies with ',k,' labels:'
        for bad_movie in [m for m,v in new_movie_labels.items() if len(v) == k]:
            print bad_movie, 'used to be:',config.ITEM_DATA[bad_movie]['labels'],'now:',new_movie_labels[bad_movie]
    

def computeExplanationStatistics(path, iterations, popularity_bin=None):
    '''
    popularity_bin contains a list of items within a specific pop-bin for each iteration
    '''
    means_and_stds = {}
    
    unexplained = 0
    skipped = 0
    counted_data_points = 0
    total_data_points = 0
    
    coverage = []
    accuracy = []
#     inverted_accuracy = []
#     odds_ratio = []
#     info_gain = []
#     inverted_info_gain = []
    length = []
    
    avg_novelty = []
    min_novelty = []
    max_novelty = []
    
    avg_surprise = []
    min_surprise = []
    max_surprise = []
    
    avg_sim_to_rec = []
    min_sim_to_rec = []
    max_sim_to_rec = []
    
#     num_of_likes = []
#     num_of_dislikes = []
#     num_of_neutrals = []
    
    overlap_w_original = []
    
    for i in range(1,iterations+1):
        
        filepath = path+str(i)+'.dat'
        with open(filepath,'rb') as data_file:
            for line in data_file:
                dataReading = line.split(':')
                
                total_data_points += 1
                
                # if popularity_bin is set, skip items that are outside of the bin 
                if popularity_bin and (dataReading[0], dataReading[2]) not in popularity_bin[i-1]:
                    skipped += 1
                    continue
                
                if dataReading[3].rstrip(' \n') == 'Cannot explain':
                    unexplained += 1
                    continue
                
                counted_data_points += 1
                
                rule = ast.literal_eval(dataReading[3])
                
                coverage.append(float(dataReading[4]))
                accuracy.append(float(dataReading[5]))
#                     inverted_accuracy.append(float(dataReading[6]))
#                     odds_ratio.append(float(dataReading[7]))
#                     info_gain.append(float(dataReading[8]))
#                     inverted_info_gain.append(float(dataReading[9]))
                assert len(rule) == int(dataReading[6])
                length.append(len(rule) / 10.0)
                
                avg_novelty.append(float(dataReading[7]))
                min_novelty.append(float(dataReading[8]))
                max_novelty.append(float(dataReading[9]))
                
                avg_surprise.append(float(dataReading[10]))
                min_surprise.append(float(dataReading[11]))
                max_surprise.append(float(dataReading[12]))
                
                avg_sim_to_rec.append(float(dataReading[13]))
                min_sim_to_rec.append(float(dataReading[14]))
                max_sim_to_rec.append(float(dataReading[15]))
                
#                     num_of_likes.append(float(dataReading[16]) / 10.0)
#                     num_of_dislikes.append(float(dataReading[17]) / 10.0)
#                     num_of_neutrals.append(float(dataReading[18]) / 10.0)
                
                overlap_w_original.append(float(dataReading[19]))
                
                # needed for t- or z-test
                means_and_stds.setdefault('coverage_full',[]).append(float(dataReading[4]))
                means_and_stds.setdefault('accuracy_full',[]).append(float(dataReading[5]))
#                     means_and_stds.setdefault('inverted_accuracy_full',[]).append(float(dataReading[6]))
#                     means_and_stds.setdefault('odds_ratio_full',[]).append(float(dataReading[7]))
#                     means_and_stds.setdefault('info_gain_full',[]).append(float(dataReading[8]))
#                     means_and_stds.setdefault('inverted_info_gain_full',[]).append(float(dataReading[9]))
                means_and_stds.setdefault('length_full',[]).append(float(len(rule) / 10.0))
                
                means_and_stds.setdefault('avg_novelty_full',[]).append(float(dataReading[7]))
                means_and_stds.setdefault('min_novelty_full',[]).append(float(dataReading[8]))
                means_and_stds.setdefault('max_novelty_full',[]).append(float(dataReading[9]))
                
                means_and_stds.setdefault('avg_surprise_full',[]).append(float(dataReading[10]))
                means_and_stds.setdefault('min_surprise_full',[]).append(float(dataReading[11]))
                means_and_stds.setdefault('max_surprise_full',[]).append(float(dataReading[12]))
                
                means_and_stds.setdefault('avg_sim_to_rec_full',[]).append(float(dataReading[13]))
                means_and_stds.setdefault('min_sim_to_rec_full',[]).append(float(dataReading[14]))
                means_and_stds.setdefault('max_sim_to_rec_full',[]).append(float(dataReading[15]))
                
#                     means_and_stds.setdefault('num_of_likes_full',[]).append(float(dataReading[16]) / 10.0)
#                     means_and_stds.setdefault('num_of_dislikes_full',[]).append(float(dataReading[17]) / 10.0)
#                     means_and_stds.setdefault('num_of_neutrals_full',[]).append(float(dataReading[18]) / 10.0)
                
                means_and_stds.setdefault('overlap_w_original_full',[]).append(float(dataReading[19]))
    
    
    print 'well, counted',counted_data_points,'data points out of total',total_data_points
    print 'unexplained',unexplained
    print 'skipped',skipped
    
    means_and_stds['coverage'] = (np.mean(coverage), np.std(coverage))
    means_and_stds['accuracy'] = (np.mean(accuracy), np.std(accuracy))
#     means_and_stds['inverted_accuracy'] = (np.mean(inverted_accuracy), np.std(inverted_accuracy))
#     means_and_stds['odds_ratio'] = (np.mean(odds_ratio), np.std(odds_ratio))
#     means_and_stds['info_gain'] = (np.mean(info_gain), np.std(info_gain))
#     means_and_stds['inverted_info_gain'] = (np.mean(inverted_info_gain), np.std(inverted_info_gain))
    means_and_stds['length'] = (np.mean(length), np.std(length))
    
    means_and_stds['avg_novelty'] = (np.mean(avg_novelty), np.std(avg_novelty))
    means_and_stds['min_novelty'] = (np.mean(min_novelty), np.std(min_novelty))
    means_and_stds['max_novelty'] = (np.mean(max_novelty), np.std(max_novelty))
    
    means_and_stds['avg_surprise'] = (np.mean(avg_surprise), np.std(avg_surprise))
    means_and_stds['min_surprise'] = (np.mean(min_surprise), np.std(min_surprise))
    means_and_stds['max_surprise'] = (np.mean(max_surprise), np.std(max_surprise))
    
    means_and_stds['avg_sim_to_rec'] = (np.mean(avg_sim_to_rec), np.std(avg_sim_to_rec))
    means_and_stds['min_sim_to_rec'] = (np.mean(min_sim_to_rec), np.std(min_sim_to_rec))
    means_and_stds['max_sim_to_rec'] = (np.mean(max_sim_to_rec), np.std(max_sim_to_rec))
    
#     means_and_stds['num_of_likes'] = (np.mean(num_of_likes), np.std(num_of_likes))
#     means_and_stds['num_of_dislikes'] = (np.mean(num_of_dislikes), np.std(num_of_dislikes))
#     means_and_stds['num_of_neutrals'] = (np.mean(num_of_neutrals), np.std(num_of_neutrals))
    
    means_and_stds['overlap_w_original'] = (np.mean(overlap_w_original), np.std(overlap_w_original))
    
    means_and_stds['unexplained'] = unexplained
    
    return means_and_stds


def compareGoodBadExplanations(path, iterations):
    
    for method in ['accuracy', 'lift', 'odds_ratio', 'info_gain', 'novelty_accuracy', 'similarity_accuracy']:
        for variation in ['_', '_extendedCandidates_', '_meanCentered_', '_meanCentered_extendedCandidates_' ]:
            
            good_stuff = computeExplanationStatistics(path+'movies_good_explanations_'+method+variation, iterations)
            bad_stuff = computeExplanationStatistics(path+'movies_bad_explanations_'+method+variation, iterations)
            
            print '\n\n---------------',method+variation,'-----------------\n'
            
            for metric in ['unexplained', 'coverage', 'accuracy', 'lift', 'odds_ratio', 'info_gain', 'length', 'avg_novelty', 'min_novelty', 'max_novelty',\
                           'avg_sim_to_rec', 'min_sim_to_rec', 'max_sim_to_rec']:
                
                if metric != 'unexplained':
                    _, p, _ = ttest_ind(good_stuff[metric+'_full'], bad_stuff[metric+'_full'])
                    if p < 0.01:
                        print p,'SIGNIFICANT!!!',metric,'good/bad:',good_stuff[metric][0],bad_stuff[metric][0]
                else:
                    print metric,'good/bad:',good_stuff[metric],bad_stuff[metric]
    


def plotExplanationChart(path, iterations, good_bad):
    '''
    plotting the explanation results for all the implemented algorithms
    '''
    
    methods = [\
               'accuracy', 'accuracy_extendedCandidates',\
               'discounted_accuracy_thresh_accfiltered', 'discounted_accuracy_thresh_accfiltered_extendedCandidates',\
               'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered', 'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered_extendedCandidates',\
               'info_gain_accfiltered', 'info_gain_accfiltered_extendedCandidates',\
               ]
#     methods = [\
#                'accuracy', 'discounted_accuracy_thresh_accfiltered',\
#                'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered',\
#                'info_gain_accfiltered',\
#                ]
#     methods = [\
#                'accuracy_extendedCandidates', 'discounted_accuracy_thresh_accfiltered_extendedCandidates',\
#                'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered_extendedCandidates',\
#                'info_gain_accfiltered_extendedCandidates',\
#                ]
    
    full_data = []
    
    for method in methods:
#         if method == 'discounted_accuracy_accfiltered_extendedCandidates':
#             data = computeExplanationStatistics(path+'movies_'+good_bad+'_explanations_'+method+variation, 1)
#         else:
#             data = computeExplanationStatistics(path+'movies_'+good_bad+'_explanations_'+method+variation, iterations)
        data = computeExplanationStatistics(path+'movies_'+good_bad+'_explanations_'+method+'_', iterations)
        full_data.append(data)
    
    width = 0.1 #0.12
    fig, ax = plt.subplots()
    
    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx()]
    
    color_list = np.arange(0,1,0.15) #0.1
    hatch_list = ['//', 'x', 'o']
    
    
    label_list = ('$overlap$', '$accuracy$', '$length/10$',\
                  '$nov_{avg}$', '$nov_{min}$', '$nov_{max}$',\
#                   '$sur_{avg}$', '$sur_{min}$', '$sur_{max}$',\
                  '$coverage$', '$sim_{avg}$', '$sim_{min}$', '$sim_{max}$')
    
    metric_list_1 = ['overlap_w_original', 'accuracy', 'length', 'avg_novelty', 'min_novelty', 'max_novelty']
#                      'avg_surprise', 'min_surprise', 'max_surprise']
    tick_list_1 = [0, 1, 2, 3, 4, 5]#, 6, 7, 8]
    
    metric_list_2 = ['coverage', 'avg_sim_to_rec', 'min_sim_to_rec', 'max_sim_to_rec']
    tick_list_2 = [6, 7, 8, 9]
    
    for counter in range(1,len(full_data)+1):
        means = [full_data[counter-1][metric][0] for metric in metric_list_1]
        if counter >= len(color_list):
            # if the number of methods is more than the number of colors, use the last color and hatch
            axes[0].bar(np.array(tick_list_1)+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)])
        else:
            # else, just use the color
            axes[0].bar(np.array(tick_list_1)+width*counter, means, width, color=str(color_list[counter-1]))
    
    axes[0].set_ylim(0.0, 1.01)
    
    
    labels = []
    for counter in range(1,len(full_data)+1):
        means = [full_data[counter-1][metric][0] for metric in metric_list_2]
        if counter >= len(color_list):
            # if the number of methods is more than the number of colors, use the last color and hatch
            labels.append(axes[1].bar(np.array(tick_list_2)+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)]) )
        else:
            # else, just use the color
            labels.append(axes[1].bar(np.array(tick_list_2)+width*counter, means, width, color=str(color_list[counter-1])) )
    
    axes[1].set_ylim(0.0, 0.15)
    
    
    # add some text for labels, title and axes ticks
    axes[0].set_ylabel('$overlap \quad accuracy \quad length/10 \quad novelty$', fontsize=14)
    axes[1].set_ylabel('$coverage \quad sim-to-rec$', fontsize=14)
    
    ax.set_xticks(np.arange(0,len(label_list),1)+len(methods)*width) #4*width
    ax.set_xticklabels( label_list, fontsize=14 )
    
    legend_methods =['accuracy', 'accuracy EX',\
                     'popularity-discounted accuracy', 'popularity-discounted accuracy EX',\
                     'uniqueness-discounted accuracy', 'uniqueness-discounted accuracy EX',\
                     'info-gain', 'info-gain EX',\
                     ]
#     legend_methods =['accuracy', 'popularity-discounted accuracy',\
#                      'uniqueness-discounted accuracy', 'info-gain',\
#                      ]
#     legend_methods =['accuracy EX', 'popularity-discounted accuracy EX',\
#                      'uniqueness-discounted accuracy EX', 'info-gain EX',\
#                      ]
    
    ax.legend( [label[0] for label in labels], legend_methods, loc=1, prop={'size':15} )
    
    plt.show()



def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out



def compareExplanationStatistics_for_Venn(explanations_folder, filenames, method_1, method_2, iterations):
    
    means_and_stds_m1 = {}
    avg_novelty_m1 = []
    min_novelty_m1 = []
    max_novelty_m1 = []
    avg_sim_m1 = []
    min_sim_m1 = []
    max_sim_m1 = []
    length_m1 = []
    acc_m1 = []
    cov_m1 = []
    
    means_and_stds_overlap = {}
    avg_novelty_overlap = []
    min_novelty_overlap = []
    max_novelty_overlap = []
    avg_sim_overlap = []
    min_sim_overlap = []
    max_sim_overlap = []
    length_overlap = []
    acc_overlap = []
    cov_overlap = []
    
    means_and_stds_m2 = {}
    avg_novelty_m2 = []
    min_novelty_m2 = []
    max_novelty_m2 = []
    avg_sim_m2 = []
    min_sim_m2 = []
    max_sim_m2 = []
    length_m2 = []
    acc_m2 = []
    cov_m2 = []
    
    avg_overlap = 0.0
    data_points = 0.0
    
    for i in range(1,iterations+1):
        
        # need to create data matrix for novelty computation
        train_filename, test_filename, user_means_filename, eval_item_filename, opinion_filename = filenames[i-1]
        train_data = trainData.TrainData(train_filename, user_means_filename)
        
        config.ITEM_OPINIONS.clear()
        with open(opinion_filename, 'rb') as opinion_file:
            config.ITEM_OPINIONS = pickle.load(opinion_file)
        
        filepath_1 = explanations_folder+'movies_good_explanations_'+method_1+'_'+str(i)+'.dat'
        filepath_2 = explanations_folder+'movies_good_explanations_'+method_2+'_'+str(i)+'.dat'
        
        with open(filepath_1,'rb') as data_file_1, open(filepath_2,'rb') as data_file_2:
            for line_1, line_2 in izip(data_file_1,data_file_2):
                dataReading_1 = line_1.split(':')
                dataReading_2 = line_2.split(':')
                
                assert dataReading_1[2] == dataReading_2[2]
                explained_item_id = dataReading_1[2]
                
                if dataReading_1[3].rstrip(' \n') == 'Cannot explain' or dataReading_2[3].rstrip(' \n') == 'Cannot explain':
                    continue
                
                rule_1 = set(ast.literal_eval(dataReading_1[3]))
                rule_2 = set(ast.literal_eval(dataReading_2[3]))
                
                rule_items_m1 = rule_1.difference(rule_2)
                rule_items_m2 = rule_2.difference(rule_1)
                rule_items_overlap = rule_1.intersection(rule_2)
                
                if len(rule_items_m1) > 0:
                    avg_nov = novelty.getListNovelty(train_data, rule_items_m1)
                    min_nov, max_nov = novelty.getMinMaxNovelty(train_data, rule_items_m1)
                    avg_novelty_m1.append(avg_nov)
                    min_novelty_m1.append(min_nov)
                    max_novelty_m1.append(max_nov)
                    length_m1.append(len(rule_items_m1)/ 10.0)
                    rule_metrics = explanationMetrics.getRuleMetrics(train_data, rule_items_m1, (explained_item_id,'like'), verbose=False)
                    acc_m1.append(rule_metrics['accuracy'])
                    cov_m1.append(rule_metrics['coverage'])
                    
                    tmp_sims = []
                    rec_labels = set(config.ITEM_DATA[explained_item_id]['labels'])
                    for i_id, _ in rule_items_m1: 
                        rule_item_labels = set(config.ITEM_DATA[i_id]['labels'])
                        label_intersection = rec_labels.intersection(rule_item_labels)
                        tmp_sims.append((i_id, float(len(label_intersection)) / (len(rec_labels) + len(rule_item_labels) - len(label_intersection)) ))
                    sorted_sims = sorted(tmp_sims, key=operator.itemgetter(1))
                    avg_sim = sum([sim for _,sim in sorted_sims]) / float(len(sorted_sims))
                    min_sim = sorted_sims[0][1]
                    max_sim = sorted_sims[-1][1]
                    avg_sim_m1.append(avg_sim)
                    min_sim_m1.append(min_sim)
                    max_sim_m1.append(max_sim)
                
                if len(rule_items_m2) > 0:
                    avg_nov = novelty.getListNovelty(train_data, rule_items_m2)
                    min_nov, max_nov = novelty.getMinMaxNovelty(train_data, rule_items_m2)
                    avg_novelty_m2.append(avg_nov)
                    min_novelty_m2.append(min_nov)
                    max_novelty_m2.append(max_nov)
                    length_m2.append(len(rule_items_m2)/ 10.0)
                    rule_metrics = explanationMetrics.getRuleMetrics(train_data, rule_items_m2, (explained_item_id,'like'), verbose=False)
                    acc_m2.append(rule_metrics['accuracy'])
                    cov_m2.append(rule_metrics['coverage'])
                    
                    tmp_sims = []
                    rec_labels = set(config.ITEM_DATA[explained_item_id]['labels'])
                    for i_id, _ in rule_items_m2: 
                        rule_item_labels = set(config.ITEM_DATA[i_id]['labels'])
                        label_intersection = rec_labels.intersection(rule_item_labels)
                        tmp_sims.append((i_id, float(len(label_intersection)) / (len(rec_labels) + len(rule_item_labels) - len(label_intersection)) ))
                    sorted_sims = sorted(tmp_sims, key=operator.itemgetter(1))
                    avg_sim = sum([sim for _,sim in sorted_sims]) / float(len(sorted_sims))
                    min_sim = sorted_sims[0][1]
                    max_sim = sorted_sims[-1][1]
                    avg_sim_m2.append(avg_sim)
                    min_sim_m2.append(min_sim)
                    max_sim_m2.append(max_sim)
                
                if len(rule_items_overlap) > 0:
                    avg_nov = novelty.getListNovelty(train_data, rule_items_overlap)
                    min_nov, max_nov = novelty.getMinMaxNovelty(train_data, rule_items_overlap)
                    avg_novelty_overlap.append(avg_nov)
                    min_novelty_overlap.append(min_nov)
                    max_novelty_overlap.append(max_nov)
                    length_overlap.append(len(rule_items_overlap)/ 10.0)
                    rule_metrics = explanationMetrics.getRuleMetrics(train_data, rule_items_overlap, (explained_item_id,'like'), verbose=False)
                    acc_overlap.append(rule_metrics['accuracy'])
                    cov_overlap.append(rule_metrics['coverage'])
                    
                    tmp_sims = []
                    rec_labels = set(config.ITEM_DATA[explained_item_id]['labels'])
                    for i_id, _ in rule_items_overlap: 
                        rule_item_labels = set(config.ITEM_DATA[i_id]['labels'])
                        label_intersection = rec_labels.intersection(rule_item_labels)
                        tmp_sims.append((i_id, float(len(label_intersection)) / (len(rec_labels) + len(rule_item_labels) - len(label_intersection)) ))
                    sorted_sims = sorted(tmp_sims, key=operator.itemgetter(1))
                    avg_sim = sum([sim for _,sim in sorted_sims]) / float(len(sorted_sims))
                    min_sim = sorted_sims[0][1]
                    max_sim = sorted_sims[-1][1]
                    avg_sim_overlap.append(avg_sim)
                    min_sim_overlap.append(min_sim)
                    max_sim_overlap.append(max_sim)
                
                avg_overlap += len(rule_1.intersection(rule_2)) / float(len(rule_2))
                data_points += 1.0
    
    means_and_stds_m1['avg_novelty'] = (np.mean(avg_novelty_m1), np.std(avg_novelty_m1))
    means_and_stds_m1['min_novelty'] = (np.mean(min_novelty_m1), np.std(min_novelty_m1))
    means_and_stds_m1['max_novelty'] = (np.mean(max_novelty_m1), np.std(max_novelty_m1))
    means_and_stds_m1['avg_sim'] = (np.mean(avg_sim_m1), np.std(avg_sim_m1))
    means_and_stds_m1['min_sim'] = (np.mean(min_sim_m1), np.std(min_sim_m1))
    means_and_stds_m1['max_sim'] = (np.mean(max_sim_m1), np.std(max_sim_m1))
    means_and_stds_m1['length'] = (np.mean(length_m1), np.std(length_m1))
    means_and_stds_m1['accuracy'] = (np.mean(acc_m1), np.std(acc_m1))
    means_and_stds_m1['coverage'] = (np.mean(cov_m1), np.std(cov_m1))
    
    means_and_stds_m2['avg_novelty'] = (np.mean(avg_novelty_m2), np.std(avg_novelty_m2))
    means_and_stds_m2['min_novelty'] = (np.mean(min_novelty_m2), np.std(min_novelty_m2))
    means_and_stds_m2['max_novelty'] = (np.mean(max_novelty_m2), np.std(max_novelty_m2))
    means_and_stds_m2['avg_sim'] = (np.mean(avg_sim_m2), np.std(avg_sim_m2))
    means_and_stds_m2['min_sim'] = (np.mean(min_sim_m2), np.std(min_sim_m2))
    means_and_stds_m2['max_sim'] = (np.mean(max_sim_m2), np.std(max_sim_m2))
    means_and_stds_m2['length'] = (np.mean(length_m2), np.std(length_m2))
    means_and_stds_m2['accuracy'] = (np.mean(acc_m2), np.std(acc_m2))
    means_and_stds_m2['coverage'] = (np.mean(cov_m2), np.std(cov_m2))
    
    means_and_stds_overlap['avg_novelty'] = (np.mean(avg_novelty_overlap), np.std(avg_novelty_overlap))
    means_and_stds_overlap['min_novelty'] = (np.mean(min_novelty_overlap), np.std(min_novelty_overlap))
    means_and_stds_overlap['max_novelty'] = (np.mean(max_novelty_overlap), np.std(max_novelty_overlap))
    means_and_stds_overlap['avg_sim'] = (np.mean(avg_sim_overlap), np.std(avg_sim_overlap))
    means_and_stds_overlap['min_sim'] = (np.mean(min_sim_overlap), np.std(min_sim_overlap))
    means_and_stds_overlap['max_sim'] = (np.mean(max_sim_overlap), np.std(max_sim_overlap))
    means_and_stds_overlap['length'] = (np.mean(length_overlap), np.std(length_overlap))
    means_and_stds_overlap['accuracy'] = (np.mean(acc_overlap), np.std(acc_overlap))
    means_and_stds_overlap['coverage'] = (np.mean(cov_overlap), np.std(cov_overlap))
    
    print data_points,'data points, avg. overlap between',method_1,'and',method_2,'is',avg_overlap / data_points
    
    return means_and_stds_m1, means_and_stds_overlap, means_and_stds_m2


def plotExplanationVennChart(explanations_folder, iterations):
    '''
    plotting the explanation results for pairs of algorithms, taking the overlapping items separately
    '''
    
    config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, explanations_folder.rstrip('/')+'_splits/')
    config.MOVIES_OR_MUSIC = 'movies'
    config.LOAD_DATA = False
    config.LOAD_OPINIONS = False
    filenames = dataPreprocessing.loadData(mode='explanations',mean_center=False)
    
    full_data = {}
    
    methods = ['accuracy','discounted_accuracy_thresh_accfiltered',\
               'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered',\
               'info_gain_accfiltered']
    
    for method_1, method_2 in itertools.combinations(methods,2):
        
        m1_data, overlap_data, m2_data = compareExplanationStatistics_for_Venn(explanations_folder, filenames, method_1, method_2, iterations)
        
        m_1 = method_1
        if method_1 == 'discounted_accuracy_thresh_accfiltered':
            m_1 = 'popularity-discounted accuracy'
        elif method_1 == 'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered':
            m_1 = 'uniqueness-discounted accuracy'
        elif method_1 == 'info_gain_accfiltered':
            m_1 = 'info-gain'
        
        m_2 = method_2
        if method_2 == 'discounted_accuracy_thresh_accfiltered':
            m_2 = 'popularity-discounted accuracy'
        elif method_2 == 'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered':
            m_2 = 'uniqueness-discounted accuracy'
        elif method_2 == 'info_gain_accfiltered':
            m_2 = 'info-gain'
        
        combination = m_1 + '_vs_' + m_2
        full_data[combination] = [m1_data, overlap_data, m2_data]
    
    width = 0.14
    
    # create all axes we need
    axes = []
    
    ax0 = plt.subplot(321)
    ax1 = ax0.twinx()
    axes.append((ax0, ax1))
    
    ax2 = plt.subplot(322)
    ax3 = ax2.twinx()
    axes.append((ax2, ax3))
    
    ax4 = plt.subplot(323)
    ax5 = ax4.twinx()
    axes.append((ax4, ax5))
    
    ax6 = plt.subplot(324)
    ax7 = ax6.twinx()
    axes.append((ax6, ax7))
    
    ax8 = plt.subplot(325)
    ax9 = ax8.twinx()
    axes.append((ax8, ax9))
    
    ax10 = plt.subplot(326)
    ax11 = ax10.twinx()
    axes.append((ax10, ax11))
    
    color_list = np.arange(0,1,0.25)
    label_list = ('$length$','$accuracy$',\
                  '$nov_{avg}$', '$nov_{min}$', '$nov_{max}$',\
                  '$sim_{avg}$', '$sim_{min}$', '$sim_{max}$',\
                  '$coverage$')
    
    metric_list_1 = ['length', 'accuracy', 'avg_novelty', 'min_novelty', 'max_novelty']
    tick_list_1 = [0, 1, 2, 3, 4]
    metric_list_2 = ['coverage', 'avg_sim', 'min_sim', 'max_sim']
    tick_list_2 = [5, 6, 7, 8]
    
    assert len(full_data) == len(axes)
    
    for combination, (ax_a, ax_b) in izip(full_data.keys(), axes):
        
        labels = []
        for counter in range(1,len(full_data[combination])+1):
            means = [full_data[combination][counter-1][metric][0] for metric in metric_list_1]
            labels.append(ax_a.bar(np.array(tick_list_1)+width*counter, means, width, color=str(color_list[counter-1])) )
        ax_a.set_ylim(0.0, 0.6)
        
        for counter in range(1,len(full_data[combination])+1):
            means = [full_data[combination][counter-1][metric][0] for metric in metric_list_2]
            ax_b.bar(np.array(tick_list_2)+width*counter, means, width, color=str(color_list[counter-1]))
        ax_b.set_ylim(0.0, 0.08)
        
        # add some text for labels, title and axes ticks
        ax_a.set_ylabel('$length/10 \quad accuracy \quad novelty$', fontsize=15)
        ax_b.set_ylabel('$coverage \quad sim-to-rec$', fontsize=15)
        
        ax_a.set_xticks(np.arange(0,len(label_list),1)+3*width)
        ax_b.set_xticklabels( label_list, fontsize=15 )
        
        legend_labels = combination.split('_vs_')
        ax_a.legend( [label[0] for label in labels], [legend_labels[0]+' only', 'overlap' ,legend_labels[1]+' only'], loc=1, prop={'size':14} )
    
    plt.show()



def plotExplanationPopBinChart(path, iterations, item_popularity_bins):
    '''
    plotting the metric results for all the implemented algorithms, binned by the explained item popularity
    '''
    
    metrics = ['overlap_w_original', 'length', 'accuracy', 'coverage', 'avg_novelty', 'avg_sim_to_rec'] 
    
    methods = [\
               'accuracy', 'accuracy_extendedCandidates',\
               'discounted_accuracy_thresh_accfiltered', 'discounted_accuracy_thresh_accfiltered_extendedCandidates',\
               'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered', 'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered_extendedCandidates',\
               'info_gain_accfiltered', 'info_gain_accfiltered_extendedCandidates',\
               ]
#     methods = [\
#                'accuracy', 'discounted_accuracy_thresh_accfiltered',\
#                'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered',\
#                'info_gain_accfiltered',\
#                ]
#     methods = [\
#                'accuracy_extendedCandidates', 'discounted_accuracy_thresh_accfiltered_extendedCandidates',\
#                'discounted_accuracy_thresh_filtered_by_better_expl_accfiltered_extendedCandidates',\
#                'info_gain_accfiltered_extendedCandidates',\
#                ]
    
    full_data = {}
    
    for metric in metrics:
        for method in methods:
            metric_data = []
            for item_popularity_bin in item_popularity_bins:
                data = computeExplanationStatistics(path+'movies_good_explanations_'+method+'_', iterations, item_popularity_bin)
                metric_data.append(data[metric])
            full_data.setdefault(metric,[]).append(metric_data)
    
    width = 0.1
    
    color_list = np.arange(0,1,0.15)
    hatch_list = ['//', 'x', 'o']
    
    label_list = ('$novelty bin 1$', '$novelty bin 2$', '$novelty bin 3$', '$novelty bin 4$', '$novelty bin 5$')
    bin_list = [0, 1, 2, 3, 4]
    
    nrows = 3
    ncols = 2
    plot_counter = 0
    
    for metric_name in full_data:
        
        plot_counter += 1
        ax = plt.subplot(nrows, ncols, plot_counter)
        
        labels = []
        for counter in range(1,len(full_data[metric_name])+1):
            means = [full_data[metric_name][counter-1][zebin][0] for zebin in bin_list]
            if counter >= len(color_list):
                # if the number of methods is more than the number of colors, use the last color and hatch
                labels.append(ax.bar(np.array(bin_list)+width*counter, means, width, color=str(color_list[-1]), hatch=hatch_list[counter-len(color_list)]) )
            else:
                # else, just use the color
                labels.append(ax.bar(np.array(bin_list)+width*counter, means, width, color=str(color_list[counter-1])) )
        
        if metric_name == 'overlap_w_original':
            ax.set_ylim(0.2, 1.01)
        elif metric_name == 'accuracy':
            ax.set_ylim(0.3, 0.85)
        elif metric_name == 'coverage':
            ax.set_ylim(0.0, 0.11)
        elif metric_name == 'length':
            ax.set_ylim(0.2, 0.5)
        elif metric_name == 'avg_novelty':
            ax.set_ylim(0.1, 0.4)
        elif metric_name == 'avg_sim_to_rec':
            ax.set_ylim(0.0, 0.1)
            
        ax.set_ylabel(metric_name, fontsize=14)
        
        ax.set_xticks(np.arange(0,len(label_list),1)+len(methods)*width)
        ax.set_xticklabels( label_list, fontsize=14 )
        
        legend_methods =['accuracy', 'accuracy EX',\
                     'popularity-discounted accuracy', 'popularity-discounted accuracy EX',\
                     'uniqueness-discounted accuracy', 'uniqueness-discounted accuracy EX',\
                     'info-gain', 'info-gain EX',\
                     ]
#     legend_methods =['accuracy', 'popularity-discounted accuracy',\
#                      'uniqueness-discounted accuracy', 'info-gain',\
#                      ]
#     legend_methods =['accuracy EX', 'popularity-discounted accuracy EX',\
#                      'uniqueness-discounted accuracy EX', 'info-gain EX',\
#                      ]
        
        if metric_name == 'coverage':
            ax.legend( [label[0] for label in labels], legend_methods, loc=2, prop={'size':13} )
    
    plt.show()




if __name__ == "__main__":
    
#     # BEYOND-ACCURACY RESULTS
#     # has to end with '_movies' or '_music' for plotting coverage
    folder_path='../beyond_accuracy_data/october_results_movies/'
     
    iterations = 5
#     feking_surprise_iterations = 5
    plot_coverage = True
     
     
#     plotFullChart(folder_path+'movies_ubnon_normalized_50', iterations, plot_coverage, feking_surprise_iterations)
    plotBaselineChart_SelectedMethods(folder_path, iterations, plot_coverage)
#     plotBaselineChart(folder_path, iterations, plot_coverage)
      
#     for method in ['baseline_', 'diversity-content_0.5_', 'diversity-ratings_0.5_', 'surprise-content_0.5_', 'surprise-ratings_0.5_','novelty_0.5_']:
#         computeDataStatistics(folder_path+algorithm+'_'+method, iterations, verbose=True)
      
    exit()
    
    
    # EXPLANATION RESULTS
    ITERATIONS = 5
    explanations_folder = '../explanation_data/movie_item_explanations_5/'
    
    
    plotExplanationChart(explanations_folder, ITERATIONS, 'good')
    
#     plotExplanationVennChart(explanations_folder, ITERATIONS)
    
    #item_popularity_bins: bin_1:[ iteration_1, iteration_1, iteration_1, iteration_1, iteration_1], bin_2:[...] ]
#     item_popularity_bins = []
#     bin_1_iterations = []
#     bin_2_iterations = []
#     bin_3_iterations = []
#     bin_4_iterations = []
#     bin_5_iterations = []
#       
#     config.LABEL_FREQUENCY_THRESHOLD = 10
#     config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, '../explanation_data/movie_item_explanations_5_splits/')
#     config.MOVIES_OR_MUSIC = 'movies'
#     config.LOAD_DATA = False
#     config.LOAD_OPINIONS = False
#     filenames = dataPreprocessing.loadData(mode='explanations',mean_center=False)
#          
#     # for each iteration, compute the data point popularity bins: [((user, item), popularity), ((user, item), popularity) )]
#     for i, (train_filename, test_filename, user_means_filename, eval_item_filename, opinion_filename) in enumerate(filenames):
#            
#         train_data = trainData.TrainData(train_filename, user_means_filename)
#         item_popularities = train_data.getPopularityDict()
#            
#         eval_datapoints = []
#         with open(eval_item_filename,'rb') as expleval_file:
#             for line in expleval_file:
#                 data = line.split('\t')
#                 user_id = data[0]
#                 item_id = data[1].rstrip('\n')
#                 eval_datapoints.append( ((user_id,item_id), item_popularities[item_id]) )
#            
#         chunks = chunkIt(sorted(eval_datapoints, key=operator.itemgetter(1), reverse=False), 5)
#         bin_1_iterations.append([t[0] for t in chunks[0]])
#         bin_2_iterations.append([t[0] for t in chunks[1]])
#         bin_3_iterations.append([t[0] for t in chunks[2]])
#         bin_4_iterations.append([t[0] for t in chunks[3]])
#         bin_5_iterations.append([t[0] for t in chunks[4]])
#         
#     item_popularity_bins = [bin_1_iterations, bin_2_iterations, bin_3_iterations, bin_4_iterations, bin_5_iterations]
#      
#     plotExplanationPopBinChart(explanations_folder, ITERATIONS, item_popularity_bins)
    
    
#     compareGoodBadExplanations(explanations_folder, ITERATIONS)
    