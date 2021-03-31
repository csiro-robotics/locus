""" Precision-Recall curves plus introspection tools. """

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *

cfg_file = open('config.yml', 'r')
cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
data_dir = cfg_params['paths']['save_dir'] + 'pr_results/' 

cfg_file = open('config.yml', 'r')
cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
log_axis = cfg_params['pr_curve']['log_axis']
introspect = cfg_params['pr_curve']['introspection_table']
test_name = 'initial_' #'rot180_'
macros = { # folder, label, colour.
    0: [test_name + '00', '00', 'red'],
    1: [test_name + '02', '02', 'blue'],
    2: [test_name + '05', '05', 'green'],
    3: [test_name + '06', '06', 'yellow'],
    4: [test_name + '07', '07', 'purple'],
    5: [test_name + '08', '08', 'grey'],
}

########################################################################################################################
EPS = []
F1s = []
F1data = []
EPdata = []
table_rows = []

for i in range(len(macros)):
    folder = macros[i][0]
    label = macros[i][1]
    colour = macros[i][2]

    _dir = data_dir + folder
    if os.path.isdir(_dir):
        table_rows.append(label)

        num_true_positive = load_pickle(_dir + '/num_true_positive.pickle')
        num_false_positive = load_pickle(_dir + '/num_false_positive.pickle')
        num_true_negative = load_pickle(_dir + '/num_true_negative.pickle')
        num_false_negative = load_pickle(_dir + '/num_false_negative.pickle')

        Precisions = []
        Recalls = []
        Accuracies = []
        nThres = len(num_true_positive)

        RP100 = 0.0
        EP = 0.0
        F1max = 0.0

        for ithThres in range(nThres):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres] 
            
            nTotalTestPlaces = nTrueNegative + nFalsePositive + nTruePositive + nFalseNegative
            
            Precision = 0.0
            Recall = 0.0
            F1 = 0.0
            Acc = (nTruePositive + nTrueNegative)/nTotalTestPlaces

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)
                F1 = 2 * Precision * Recall * (1/(Precision + Recall))
            
            Precisions.append(Precision)
            Recalls.append(Recall)
            Accuracies.append(Acc)

            if F1 > F1max:
                F1max = F1
                f1max_tn = nTrueNegative 
                f1max_fp = nFalsePositive
                f1max_tp = nTruePositive 
                f1max_fn = nFalseNegative
                f1max_id = ithThres
                f1max_total = nTotalTestPlaces

            if int(Precision) == 1:
                RP100 = Recall
                EP_id = ithThres
                rp100_tn = nTrueNegative 
                rp100_fp = nFalsePositive
                rp100_tp = nTruePositive 
                rp100_fn = nFalseNegative
                rp100_total = nTotalTestPlaces

        if RP100 == 0.0:
            EP = Precisions[1]/2.0
            
        else:
            EP = 0.5 + (RP100/2.0)

        if log_axis:
            Precisions = 1- np.asarray(Precisions)
            Recalls = 1- np.asarray(Recalls)

        
        print('EP: ' , EP)
        print('F1max: ' , F1max)
        print('f1max_id: ' , f1max_id)
        EPS.append(EP)
        F1s.append(F1max)
        F1data.append([str(val) for val in (f1max_tn, f1max_tp, f1max_fn, f1max_fp, f1max_total, "{:.3f}".format(F1max))])
        EPdata.append([str(val) for val in (rp100_tn, rp100_tp, rp100_fn, rp100_fp, rp100_total, "{:.3f}".format(EP))])

        label = label + ', EP: ' + "{:.3f}".format(EP) + ', F1max: ' + "{:.3f}".format(F1max) 

        plt.plot(Recalls, Precisions, marker='.', color=colour, label=label)

##########################################################################################################################
""" Plot Precision-Recall curves """

plt.legend(prop=font_legend)
plt.title('Locus performance on KITTI')

if log_axis:
    plt.xlabel('Recall (log)', fontdict=font)
    plt.ylabel('Precision (log)', fontdict=font)
    plt.yscale('log')
    plt.xscale('log')
    ax = plt.gca()
    ax.set_xlim(1, 0.001)
    ax.set_ylim(1, 0.001)
    plt.xticks([1,0.1,0.01,0.001],['0%', '90%', '99%', '99.9%'])
    plt.yticks([1,0.1,0.01,0.001],['0%', '90%', '99%', '99.9%'])
    plt.grid(True, which='major')
    plt.grid(True, which='minor', color = 'whitesmoke')
    plt.show()
else:
    plt.xlabel('Recall', fontdict=font)
    plt.ylabel('Precision', fontdict=font)
    plt.axis([0, 1, 0, 1.1])
    plt.xticks(np.arange(0, 1.01, step=0.1)) 
    plt.grid(True)
    plt.show()

##########################################################################################################################
""" Tables for introspection of 'TN', 'TP', 'FN', 'FP' counts. """

if introspect:
    # rows = [macros[i][1] for i in range(len(macros))]
    F1columns = ('TN', 'TP', 'FN', 'FP', 'Total', 'F1max')
    EPcolumns = ('TN', 'TP', 'FN', 'FP', 'Total', 'EP')
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    table1 = ax1.table(cellText=F1data, rowLabels=table_rows, colLabels=F1columns, loc='center')
    table1.scale(0.7,2)
    ax1.axis('off')
    ax2 = fig.add_subplot(2,1,2)
    table2 = ax2.table(cellText=EPdata, rowLabels=table_rows, colLabels=EPcolumns, loc='center')
    table2.scale(0.7,2)
    ax2.axis('off')
    plt.title('Introspection table (Top: F1max, Bot: RP100)')
    plt.show()