import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

from utils.node import Neighborhood
from utils import readdata


bar_order = ['RND', 'SW', 'VAR', 'IBS', 'NBS', 'FCV', 'VBS', 'BTVR', 'ITVR', 'GCS']
color_dict = {'RND': 'C0', 'SW': 'C1', 'VAR': 'C2', 'IBS': 'C3', 'NBS': 'C4', 'FCV': 'C5', 'VBS': 'C6',
              'BTVR': 'C7', 'ITVR': 'C8', 'GCS': 'C9'}
budget_labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%']

rootdir = 'C:/Users/CnrKmrl/Documents/workbench/experiments/20171028/lasso/wunderground-US/'
topology = 'lasso'


def printMeanErrors(error_ind):
    # error_ind = 5
    mean_mae_root_path = '{modeldir}/{selection_method}/{obsrate}/errors/meanMAE_activeInf_' + \
                         'model={model}_topology={topology}_window=12_T=12_obsRate={obsrate}_trial=mean.csv'
    stderr_mae_root_path = '{modeldir}/{selection_method}/{obsrate}/errors/stderrMAE_activeInf_' + \
                           'model={model}_topology={topology}_window=12_T=12_obsRate={obsrate}_' \
                           'trial=mean.csv'
    obsRateVec = np.arange(.0, .7, .1)

    mean_mae_dict = dict()
    stderr_mae_dict = dict()
    for modelName in os.listdir(rootdir):
        if not os.path.isdir(rootdir + modelName):
            break
        for methodName in os.listdir(rootdir + modelName + os.sep):
            if 'debug' in methodName or 'old' in methodName:  # or 'err' in methodName:
                continue
            current_mean_mae = np.empty(shape=(7, 12))
            current_stderr_mae = np.zeros(shape=(7, 12))
            current_mean_over_time = np.empty(shape=(7,))
            current_stderr_over_time = np.empty(shape=(7,))
            for j in range(0, 7):
                curr = np.loadtxt(rootdir + mean_mae_root_path.format(modeldir=modelName, selection_method=methodName,
                                                                      model=modelName.lower(), topology=topology,
                                                                      obsrate=obsRateVec[j]), delimiter=',')
                current_mean_mae[j] = curr[:, error_ind]
                current_mean_over_time[j] = np.mean(current_mean_mae[j])
                curr = np.loadtxt(rootdir + stderr_mae_root_path.format(modeldir=modelName, selection_method=methodName,
                                                                        model=modelName.lower(), topology=topology,
                                                                        obsrate=obsRateVec[j]), delimiter=',')
                current_stderr_mae[j] = curr[:, error_ind]
                current_stderr_over_time[j] = np.mean(current_stderr_mae[j])
            mean_mae_dict[(modelName, methodName)] = current_mean_over_time
            stderr_mae_dict[(modelName, methodName)] = current_stderr_over_time
    meanErrorPrintOut = np.empty(shape=(len(mean_mae_dict), 9), dtype=object)
    stderrErrorPrintOut = np.empty(shape=(len(mean_mae_dict), 9), dtype=object)
    for i in range(len(mean_mae_dict)):
        current_key = mean_mae_dict.keys()[i]
        meanErrorPrintOut[i, 0] = str(current_key[0])
        meanErrorPrintOut[i, 1] = str(current_key[1])
        meanErrorPrintOut[i, 2:] = mean_mae_dict[current_key]
        stderrErrorPrintOut[i, 0] = str(current_key[0])
        stderrErrorPrintOut[i, 1] = str(current_key[1])
        stderrErrorPrintOut[i, 2:] = stderr_mae_dict[current_key]

    if 4 == error_ind:
        np.savetxt(rootdir + 'meanMAE.csv', meanErrorPrintOut, delimiter=',', fmt='%s')
        np.savetxt(rootdir + 'stderrMAE.csv', stderrErrorPrintOut, delimiter=',', fmt='%s')
    elif 5 == error_ind:
        np.savetxt(rootdir + 'meanMSE.csv', meanErrorPrintOut, delimiter=',', fmt='%s')
        np.savetxt(rootdir + 'stderrMSE.csv', stderrErrorPrintOut, delimiter=',', fmt='%s')


def printEvidenceEntropies():
    evid_root_path = '{modeldir}/{selection_method}/{obsrate}/evidences/'
    obs_rate_vec = np.arange(.1, .7, .1)
    entropy_dict = dict()
    for modelName in os.listdir(rootdir):
        if not os.path.isdir(rootdir + modelName):
            break
        for methodName in os.listdir(rootdir + modelName + os.sep):
            if 'debug' in methodName or 'old' in methodName:  # or 'err' in methodName:
                continue
            entropy_dict[(modelName, methodName)] = np.zeros(shape=obs_rate_vec.shape[0] + 2, dtype=object)
            for j in range(1, 7):
                current_dir = rootdir + evid_root_path.format(modeldir=modelName, selection_method=methodName,
                                                              obsrate=obs_rate_vec[j-1])
                avg_entropy = 0
                file_count = 0
                for f in os.listdir(current_dir):
                    current_evid = np.loadtxt(current_dir + f, delimiter=',')
                    avg_entropy += entropy(np.sum(current_evid, axis=1), base=2)
                    file_count += 1
                avg_entropy /= file_count
                entropy_dict[(modelName, methodName)][:2] = [modelName, methodName]
                entropy_dict[(modelName, methodName)][2 + j - 1] = avg_entropy
    evid_print_out = np.zeros(shape=(len(entropy_dict.keys()), obs_rate_vec.shape[0] + 2), dtype=object)
    for i in range(len(entropy_dict)):
        evid_print_out[i] = entropy_dict[entropy_dict.keys()[i]]
    np.savetxt(rootdir + 'evidenceEntropy.csv', evid_print_out, delimiter=',', fmt='%s')


def writeFileBarPlotsEntropies():
    entropies = pd.read_csv(rootdir + 'evidenceEntropy.csv', header=None, index_col=[0, 1], names=budget_labels[1:])
    offset = -.5
    barwidth = .1
    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for idx in sorted(entropies.index.get_values(), key=lambda x: bar_order.index(x[1])):
        ax.bar(np.arange(len(entropies.columns)) * 1.8 + offset, entropies.loc[idx].get_values(), barwidth,
                label='-'.join(idx), color=color_dict[idx[1]])
        offset += barwidth + .05
    ax.set_xticks(np.arange(len(entropies.columns)) * 1.8)
    ax.set_xticklabels(entropies.columns)
    # ax.set_xticks(np.arange(len(entropies.columns[1:])) * 1.5, entropies.columns[1:])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Entropy of evidence selection\nWeather Underground US JAN/FEB data with Lasso')
    plt.xlabel('Rate of observed sensors')
    plt.ylabel('Entropy of selections')
    plt.grid(linestyle='dotted', axis='y', zorder=0)
    # plt.show(bbox_inches='tight')
    fig.savefig(rootdir + 'entropyBars.png', bbox_inches='tight', dpi=300)


def writeFileBarPlots_MAE_MSE():
    meanMAE = pd.read_csv(rootdir + 'meanMAE.csv', header=None, index_col=[0, 1], names=budget_labels)
    stderrMAE = pd.read_csv(rootdir + 'stderrMAE.csv', header=None, index_col=[0, 1], names=budget_labels)
    offset = -.5
    barwidth = .1
    # plt.plot([.1, 5.7], [3.5, 3.5], color='gray', linestyle='dotted', zorder=0)
    for idx in sorted(meanMAE.index.get_values(), key=lambda x: bar_order.index(x[1])):
        plt.bar(np.arange(len(meanMAE.columns[1:])) * 1.8 + offset, meanMAE.loc[idx].get_values()[1:], barwidth,
                label='-'.join(idx), yerr=stderrMAE.loc[idx].get_values()[1:], color=color_dict[idx[1]])
        offset += barwidth + .05

    plt.xticks(np.arange(len(meanMAE.columns[1:])) * 1.8, meanMAE.columns[1:])
    plt.legend()
    plt.title('MAE on all sensors\nWeather Underground US JAN/FEB data with Lasso')
    plt.xlabel('Rate of observed sensors')
    plt.ylabel('MAE')
    plt.grid(linestyle='dotted', axis='y', zorder=0)
    plt.savefig(rootdir + 'MAEonAll.jpg', dpi=300)
    plt.clf()

    meanMSE = pd.read_csv(rootdir + 'meanMSE.csv', header=None, index_col=[0, 1], names=budget_labels)
    stderrMSE = pd.read_csv(rootdir + 'stderrMSE.csv', header=None, index_col=[0, 1], names=budget_labels)
    offset = -.5
    plt.grid(linestyle='dotted', axis='y', zorder=-1)

    for idx in sorted(meanMSE.index.get_values(), key=lambda x: bar_order.index(x[1])):
        plt.bar(np.arange(len(meanMSE.columns[1:])) * 1.8 + offset, meanMSE.loc[idx].get_values()[1:], barwidth,
                label='-'.join(idx), yerr=stderrMSE.loc[idx].get_values()[1:], color=color_dict[idx[1]])
        offset += barwidth + .05

    plt.xticks(np.arange(len(meanMSE.columns[1:])) * 1.8, meanMSE.columns[1:])
    plt.legend()
    plt.title('MSE on all sensors\nWeather Underground US JAN/FEB data with Lasso')
    plt.xlabel('Rate of observed sensors')
    plt.ylabel('MSE')
    plt.savefig(rootdir + 'MSEonAll.jpg', dpi=300)


def generateSumVarianceBars():
    var_path = '{modeldir}/{selection_method}/{obsrate}/predictions/var/'
    # var_file_name = 'predResults_activeInf_model={model}_T=12_trial={trial}_obsRate={obsrate}'
    obsRateVec = np.arange(.0, .7, .1)
    var_dict = dict()
    for modelName in os.listdir(rootdir):
        if not os.path.isdir(rootdir + modelName):
            break
        for methodName in os.listdir(rootdir + modelName + os.sep):
            if 'debug' in methodName or 'old' in methodName:  # or 'err' in methodName:
                continue
            var_dict[(modelName, methodName)] = list()
            for obsrate in obsRateVec:
                sums_of_var = list()
                current_dir = rootdir + var_path.format(modeldir=modelName, selection_method=methodName,
                                                        obsrate=obsrate)
                for f in os.listdir(current_dir):
                    varmat = np.loadtxt(current_dir + f, delimiter=',')
                    varmat[varmat == -1] = 0
                    if np.any(varmat < 0):
                        print methodName, obsrate, f
                        print varmat[varmat < 0]
                    sums_of_var.append(varmat.sum())
                var_dict[(modelName, methodName)].append(np.mean(sums_of_var))  # / ((1-obsrate) * varmat.shape[0]))
    resultdf = pd.DataFrame(var_dict, index=obsRateVec).T
    resultdf.columns = budget_labels
    offset = -.35
    barwidth = .1
    plt.clf()
    for idx in sorted(resultdf.index.get_values(), key=lambda x: bar_order.index(x[1])):
        plt.bar(np.arange(len(resultdf.columns[1:]))*1.8 + offset, resultdf.loc[idx].get_values()[1:], barwidth,
                label='-'.join(idx), zorder=3, color=color_dict[idx[1]])
        offset += barwidth + .05
    plt.xticks(np.arange(len(resultdf.columns[1:]))*1.8, resultdf.columns[1:])
    plt.legend()
    plt.title('Sum of variances\nWeather Underground US JAN/FEB data with Lasso')
    plt.xlabel('Rate of observed sensors')
    plt.ylabel('Sum of variances')
    plt.grid(linestyle='dotted', axis='y', zorder=0)
    plt.savefig(rootdir + 'totalVar.jpg', dpi=300)


def checkNegativeVariance():
    rootdir = 'C:/Users/CnrKmrl/Documents/workbench/experiments/20170822/k2_bin5/wunderground-IL/'
    var_path = '{modeldir}/{selection_method}/{obsrate}/predictions/var/'
    var_file_name = 'predResults_activeInf_model={model}_T=12_trial={trial}_obsRate={obsrate}.csv'
    obsRateVec = np.arange(.1, .7, .1)
    modelName = 'DGBN'
    methodName = 'VAR'
    for obsrate in obsRateVec:
        for trial in range(5):
            fname = rootdir + var_path.format(modeldir=modelName, selection_method=methodName, obsrate=obsrate) + \
            var_file_name.format(model=modelName.lower(), trial=trial, obsrate=obsrate)
            varmat = np.loadtxt(fname, delimiter=',')
            varmat[varmat == -1] = 0
            if np.any(varmat < -0.5):
                print fname, varmat.min(), np.unravel_index(varmat.argmin(), varmat.shape)


def plotPredictionsAgainstGroundTruth():
    trainset, testset = readdata.convert_time_window_df_randomvar_hour(True,
                                                        Neighborhood.itself_previous_others_current)
    testset = testset[:, 22:34]
    testmat = np.vectorize(lambda x: x.true_label)(testset)
    obsrates = np.arange(.0, .55, .1)
    trials = range(1)
    prediction_file_path = r'C:\Users\CnrKmrl\Documents\workbench\experiments\20171028\lasso\temperature\DGBN\SW\{obsrate}\predictions\mean\predResults_activeInf_model=dgbn_T=12_trial={trial}_obsRate={obsrate}.csv'

    predictions = [None] * obsrates.shape[0]
    predictions[0] = np.loadtxt(prediction_file_path.format(obsrate='0.0', trial=0), delimiter=',')
    for obsrate in obsrates[1:]:
        prediction_on_trials = [None] * len(trials)
        for trial in trials:
            prediction_on_trials[trial] = np.loadtxt(prediction_file_path.format(obsrate=obsrate, trial=trial),
                                                     delimiter=',')
        predictions[int(obsrate*10)] = np.array(prediction_on_trials).mean(axis=0)

    # plt.plot(testmat.mean(axis=0), label='true_values')
    # for prediction, obsrate in zip(predictions, obsrates):
    #     plt.plot(prediction.mean(axis=0), label=obsrate)
    # plt.title('True values vs predictions with GCS\non Intel temperature test data')
    # plt.xlabel('Time slices')
    # plt.ylabel('Temperature')
    # plt.legend()
    # # plt.show()
    # plt.savefig(r'C:\Users\CnrKmrl\Documents\workbench\experiments\20171028\lasso\temperature\predictedMeansVSTrueOnTest_GCS.jpg', dpi=300)

    plt.plot(testmat.std(axis=0), label='true_values')
    for prediction, obsrate in zip(predictions, obsrates):
        plt.plot(prediction.std(axis=0), label=obsrate)
    plt.title('Std dev of true values over variables vs std dev of\npredictions over variables with SW\non Intel temperature test data')
    plt.xlabel('Time slices')
    plt.ylabel('Temperature')
    plt.legend()
    # plt.show()
    plt.savefig(r'C:\Users\CnrKmrl\Documents\workbench\experiments\20171028\lasso\temperature\stddevOfPredictionsVSTrueOnTest_SW.jpg', dpi=300)

def plotVarPredictions():
    obsrates = np.arange(.0, .55, .1)
    trials = range(1)
    prediction_file_path = r'C:\Users\CnrKmrl\Documents\workbench\experiments\20171028\lasso\temperature\DGBN\GCS\{obsrate}\predictions\var\predResults_activeInf_model=dgbn_T=12_trial={trial}_obsRate={obsrate}.csv'

    predictions = [None] * obsrates.shape[0]
    predictions[0] = np.loadtxt(prediction_file_path.format(obsrate='0.0', trial=0), delimiter=',')
    for obsrate in obsrates[1:]:
        prediction_on_trials = [None] * len(trials)
        for trial in trials:
            prediction_on_trials[trial] = np.loadtxt(prediction_file_path.format(obsrate=obsrate, trial=trial),
                                                     delimiter=',')
        predictions[int(obsrate*10)] = np.array(prediction_on_trials).mean(axis=0)

    plt.plot([])
    for prediction, obsrate in zip(predictions, obsrates):
        plt.plot(prediction.mean(axis=0), label=obsrate)
    plt.title('Predicted variances GCS\non Intel temperature test data')
    plt.xlabel('Time slices')
    plt.ylabel('Variance of Temperature (averaged over sensors)')
    plt.legend()
    # plt.show()
    plt.savefig(r'C:\Users\CnrKmrl\Documents\workbench\experiments\20171028\lasso\temperature\predictedVarsVSTrueOnTest_GCS.jpg', dpi=300)


if __name__ == '__main__':
    # plotPredictionsAgainstGroundTruth()

    printMeanErrors(4)
    printMeanErrors(5)
    writeFileBarPlots_MAE_MSE()
    printEvidenceEntropies()
    writeFileBarPlotsEntropies()
    generateSumVarianceBars()

    # map(lambda x: '-'.join(x), df.index.get_values())


    # for i in model_indices:
    #     for j in range(0, 7):
    #         curr = np.loadtxt(mean_mae_root_path.format(modeldir=model_dirs[i], selection_method=selection_method,
    #                                                     model=model_names[i], topology=topology,
    #                                                     obsrate=obsRateVec[j]), delimiter=',')
    #         mean_mae[i, j] = curr[:, error_ind]
    #         curr = np.loadtxt(stderr_mae_root_path.format(modeldir=model_dirs[i], selection_method=selection_method,
    #                                                       model=model_names[i],topology=topology,
    #                                                       obsrate=obsRateVec[j]), delimiter=',')
    #         stderr_mae[i, j] = curr[:, error_ind]
    #     mean_over_time[i] = np.mean(mean_mae[i], axis=1)
    #
    # np.savetxt(root_dir + 'compare4_noGP_obsIncluded.csv', mean_over_time, delimiter=', ')
    #
    # n = 12
    # figsize = (15.5, 7)
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # bar_width = 0.8  # default: 0.8
    # bar_locations = np.arange(n)*bar_width*6
    # barList = list()
    # mycolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for i in range(0, 7):
    #     # bar0 = ax.bar(bar_locations, mean_mae_gp[i, :], bar_width, yerr=stderr_mae_gp[i, :], label='Gaussian Process')
    #     bar1 = ax.bar(bar_locations+0*bar_width, mean_mae_rnd[i, :], bar_width, color='r',
    #                   yerr=stderr_mae_rnd[i, :], ecolor='r', label='gDBn_RND')
    #     bar2 = ax.bar(bar_locations+1*bar_width, mean_mae_sw[i, :], bar_width, color='g',
    #                   yerr=stderr_mae_sw[i, :], ecolor='g', label='dGBn_SW')
    #     bar3 = ax.bar(bar_locations+2*bar_width, mean_mae_ibs[i, :], bar_width, color='c',
    #                   yerr=stderr_mae_ibs[i, :], label='dGBn_IBS')
    #     bar4 = ax.bar(bar_locations+3*bar_width, mean_mae_nbs[i, :], bar_width, color='m',
    #                   yerr=stderr_mae_nbs[i, :], label='dGBn_NBS')
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Prediction Methods')
    #     ax.grid(axis='y')
    #     ax.set_xticks(bar_locations+bar_width)
    #     ax.set_xticklabels(np.arange(n))
    #     # ax.set_yticks(np.arange(.0,6.0,.5))
    #     plt.title('Comparison of Random Selection, Sliding Window and Impact Based Selection'
    #               '\nby Mean Absolute Error on evidence rate {}% - Including Observation'.
    #               format(str(int(obsRateVec[i]*100))))
    #     plt.xlabel('Time slices')
    #     plt.ylabel('Mean Absolute Error')
    #     plt.subplots_adjust(right=.8)
    #     # plt.show()
    #     plt.savefig(root_dir + 'RNDvsSWvsIBSvsNBS_obsIncl/' +
    #                 'RNDvsSWvsIBSvsNBS_unobsOnly_obsrate={}percent'.format(
    #                     str(int(obsRateVec[i]*100))))
    #     plt.cla()
