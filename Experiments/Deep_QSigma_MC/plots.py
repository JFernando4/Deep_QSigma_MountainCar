import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval

SAMPLE_SIZE = 100
NUMBER_OF_EPISODES = 500


def retrieve_method_return_data(results_path, method_name):

    method_results_path = os.path.join(results_path, method_name)
    return_data = np.zeros(shape=(SAMPLE_SIZE, NUMBER_OF_EPISODES), dtype=np.float64)
    for i in range(SAMPLE_SIZE):
        agent_folder = os.path.join(method_results_path, 'agent_'+ str(i + 1))
        agent_results_filepath = os.path.join(agent_folder, 'results.p')
        with open(agent_results_filepath, mode='rb') as results_file:
            agent_data = pickle.load(results_file)
            agent_return_data = agent_data['return_per_episode']
        return_data[i] = agent_return_data

    return return_data


def compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors):
    for i in range(len(method_names)):
        method_name = method_names[i]
        return_data = retrieve_method_return_data(results_path, method_name)
        avg = np.average(return_data, axis=0)
        std = np.std(return_data, axis=0, ddof=1)
        upper_ci, lower_ci, _ = compute_tdist_confidence_interval(avg, std, 0.05, SAMPLE_SIZE)
        method_data[method_name]['return_data'] = return_data
        method_data[method_name]['avg'] = avg
        method_data[method_name]['std'] = std
        method_data[method_name]['uci'] = upper_ci
        method_data[method_name]['lci'] = lower_ci
        method_data[method_name]['color'] = colors[i]
        method_data[method_name]['shade_color'] = shade_colors[i]


def plot_avg_return_per_episode(methods_data, ylim=(0, 1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name'):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES)+1

    for name in methods_data.keys():
        plt.plot(x, methods_data[name]['avg'], color=methods_data[name]['color'], linewidth=1)
        plt.fill_between(x, methods_data[name]['lci'], methods_data[name]['uci'],
                         color=methods_data[name]['shade_color'])

    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def plot_cummulative_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name'):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES) + 1

    for name in methods_data.keys():
        cum_average = np.divide(np.cumsum(methods_data[name]['avg']), x)
        plt.plot(x, cum_average, color=methods_data[name]['color'], linewidth=1)


    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def plot_moving_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name',
                        window=30):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES - window) + window

    for name in methods_data.keys():
        plt.plot(np.arange(NUMBER_OF_EPISODES) + 1, methods_data[name]['avg'],
                 color=methods_data[name]['shade_color'], linewidth=1)

    for name in methods_data.keys():
        moving_average = np.zeros(NUMBER_OF_EPISODES - window , dtype=np.float64)
        index = 0
        for i in range(NUMBER_OF_EPISODES - window):
            moving_average[i] += np.average(methods_data[name]['avg'][index:index+window])
            index += 1
        plt.plot(x, moving_average, color=methods_data[name]['color'], linewidth=1.3)

    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def format_data_for_interval_average_plot(methods_data, interval_size):
    plot_data = {}
    for name in methods_data.keys():
        plot_data[name] = {}
        data_points = int(NUMBER_OF_EPISODES / interval_size)
        interval_avg_data = np.zeros(shape=data_points, dtype=np.float64)
        interval_std_data = np.zeros(shape=data_points, dtype=np.float64)
        interval_me_data = np.zeros(shape=data_points, dtype=np.float64)

        index = 0
        for i in range(int(NUMBER_OF_EPISODES / interval_size)):
            # avg across (interval_size) episodes
            data_at_interval = np.average(methods_data[name]['return_data'][:, index:index + interval_size], axis=1)
            interval_avg = np.average(data_at_interval)  # avg across (sample_size) runs
            interval_std = np.std(data_at_interval, ddof=1)  # std across (sample_size) runs
            interval_uci, interval_lci, margin_of_error = compute_tdist_confidence_interval(interval_avg,
                                                                                            interval_std, 0.05,
                                                                                            SAMPLE_SIZE)
            data_index = int(index / interval_size)
            interval_avg_data[data_index] = interval_avg
            interval_std_data[data_index] = interval_std
            interval_me_data[data_index] = margin_of_error
            index += interval_size
        plot_data[name]['avg'] = interval_avg_data
        plot_data[name]['std'] = interval_std_data
        plot_data[name]['me'] = interval_me_data
    return plot_data


def plot_interval_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name',
                          interval_size=50, xticks=True, yticks=True, shaded=False):
    assert NUMBER_OF_EPISODES % interval_size == 0
    assert isinstance(methods_data, dict)

    plot_data = format_data_for_interval_average_plot(methods_data, interval_size)
    x = (np.arange(int(NUMBER_OF_EPISODES / interval_size)) + 1) * interval_size

    for name in plot_data.keys():
        if shaded:
            plt.plot(x, plot_data[name]['avg'], color=method_data[name]['color'])
            plt.fill_between(x, plot_data[name]['avg'] - plot_data[name]['me'],
                             plot_data[name]['avg'] + plot_data[name]['me'],
                             color=methods_data[name]['shade_color'])
        else:
            plt.errorbar(x, plot_data[name]['avg'], yerr=plot_data[name]['me'], color=methods_data[name]['color'])

    plt.tick_params(axis='both', which='major', labelsize=14)
    if not xticks:
        locs, _ = plt.xticks()
        plt.xticks(locs, [])
    if not yticks:
        locs, _ = plt.yticks()
        plt.yticks(locs, [])

    plt.xlim([-10, NUMBER_OF_EPISODES+10])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig('Plots/' + figure_name, dpi=200)
    plt.close()


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    """ Experiment Types """
    parser.add_argument('-offpolicy', action='store_true', default=False)
    parser.add_argument('-annealing', action='store_true', default=False)
    parser.add_argument('-ESvsQL', action='store_true', default=False)
    parser.add_argument('-nstep', action='store_true', default=False)
    parser.add_argument('-best_nstep', action='store_true', default=False)
    parser.add_argument('-ds_comparison', action='store_true', default=False)
    parser.add_argument('-tnetwork_comparison', action='store_true', default=False)
    """ Plot Types """
    parser.add_argument('-cumulative_avg', action='store_true', default=False)
    parser.add_argument('-moving_avg', action='store_true', default=False)
    parser.add_argument('-avg_per_episode', action='store_true', default=False)
    parser.add_argument('-interval_avg', action='store_true', default=False)
    args = parser.parse_args()


    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")
    sample_size = 100
    # std = standard deviation, avg = average, uci = upper confidence interval, lci = lower confidence interval
    # me = margin of error

    ##########################################
    """ On-Policy vs Off-Policy Experiment """
    ##########################################
    if args.offpolicy:
        """ Experiment Colors """
        colors = ['#025D8C',    # Blue      - Off-Policy
                  '#FBB829']    # Yellow    - On-Policy

        shade_colors = ['#b3cddb',  # Light Blue      - Off-Policy
                        '#ffe7b1']  # Light Yellow    - On-Policy

        # Sarsaparser.add_argument('-ESvsQL', action='store_true', default=False)
        method_names = ['Sarsa_OffPolicy', 'Sarsa_OnPolicy']
        method_data = {'Sarsa_OffPolicy': {},
                       'Sarsa_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1300, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='Sarsa_On_vs_Off')

        # Q(0.5)
        method_names = ['QSigma0.5_OffPolicy', 'QSigma0.5_OnPolicy']
        method_data = {'QSigma0.5_OffPolicy': {},
                       'QSigma0.5_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1300, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='QSigma05_On_vs_Off')

        # Decaying Sigma
        method_names = ['DecayingSigma_OffPolicy', 'DecayingSigma_OnPolicy']
        method_data = {'DecayingSigma_OffPolicy': {},
                       'DecayingSigma_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1300, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='DecayingSigma_On_vs_Off')

    #################################
    """ Annealing vs No Annealing """
    #################################
    if args.annealing:
        """ Experiment Colors """
        colors = ['#7FAF1B',    # Green - Annealing
                  '#A80000']    # Red   - No Annealing

        shade_colors = ['#d5e4b3',  # Green - Annealing
                        '#e6b3b3']  # Red   - No Annealing

        # Sarsa
        method_names = ['Sarsa_wAnnealingEpsilon_wOnlineBprobabilities', 'Sarsa_OnPolicy']
        method_data = {'Sarsa_wAnnealingEpsilon_wOnlineBprobabilities': {},
                       'Sarsa_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1000, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='Sarsa_Annealing_vs_NoAnnealing')


        # Q(0.5)
        method_names = ['QSigma0.5_wAnnealingEpsilon_wOnlineBprobabilities', 'QSigma0.5_OnPolicy']
        method_data = {'QSigma0.5_wAnnealingEpsilon_wOnlineBprobabilities': {},
                       'QSigma0.5_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1000, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='QSigma05_Annealing_vs_NoAnnealing')

        # Expected Sarsa
        method_names = ['ExpectedSarsa_wAnnealingEpsilon', 'ExpectedSarsa']
        method_data = {'ExpectedSarsa_wAnnealingEpsilon': {},
                       'ExpectedSarsa': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1000, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='ExpectedSarsa_Annealing_vs_NoAnnealing')

        # QLearning
        method_names = ['QLearning_wAnnealingEpsilon', 'QLearning']
        method_data = {'QLearning_wAnnealingEpsilon': {},
                       'QLearning': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1000, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='QLearning_Annealing_vs_NoAnnealing')

        # Decaying Sigma
        method_names = ['DecayingSigma_wAnnealingEpsilon_wOnlineBprobabilities', 'DecayingSigma_OnPolicy']
        method_data = {'DecayingSigma_wAnnealingEpsilon_wOnlineBprobabilities': {},
                       'DecayingSigma_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(method_data, ylim=(-1000, 0), ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='DecayingSigma_Annealing_vs_NoAnnealing')

    ##########################################
    """ Extra: QLearning vs Expected Sarsa """
    ##########################################
    if args.ESvsQL:
        """ Experiment Colors """
        colors = ['#FF0066',    # Hot Pink      - Expected Sarsa
                  '#C0ADDB']    # Purple        - QLearning

        shade_colors = ['#ffcce1',  # Hot Pink     - Expected Sarsa
                        '#e8e0f2']  # Purple       - QLearning

        method_names = ['ExpectedSarsa', 'QLearning']
        method_data = {'ExpectedSarsa': {},
                       'QLearning': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-600, -100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='ExpectedSarsa_vs_QLearning',
                                  interval_size=50)

    ##########################
    """ n-Step Experiments """
    ##########################
    if args.nstep:
        """ Colors """
        colors = ['#CAE8A2',    # Green     - n = 1
                  '#F0D878',    # Yellow    - n = 3
                  '#FBB829',    # Orange    - n = 5
                  '#FE4365',    # Pink      - n = 10
                  '#A40802']    # Red       - n = 20

        shade_colors = ['#', '#', '#', '#', '#']

        # Sarsa #
        method_names = ['Sarsa_OnPolicy', 'Sarsa_n3', 'Sarsa_n5', 'Sarsa_n10', 'Sarsa_n20']
        method_data = {'Sarsa_OnPolicy': {},
                       'Sarsa_n3': {},
                       'Sarsa_n5': {},
                       'Sarsa_n10': {},
                       'Sarsa_n20': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-1500, -100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='nStep_Sarsa',
                                  interval_size=50)

        # QSigma0.5 #
        method_names = ['QSigma0.5_OnPolicy', 'QSigma0.5_n3', 'QSigma0.5_n5', 'QSigma0.5_n10', 'QSigma0.5_n20']
        method_data = {'QSigma0.5_OnPolicy': {},
                       'QSigma0.5_n3': {},
                       'QSigma0.5_n5': {},
                       'QSigma0.5_n10': {},
                       'QSigma0.5_n20': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-1500, -100),
                                  ytitle='Average Return Over the Previous 50 Episodes',
                                  xtitle='Episode Number', figure_name='nStep_QSigma05',
                                  interval_size=50)

        # TreeBackup #
        method_names = ['ExpectedSarsa', 'TreeBackup_n3', 'TreeBackup_n5', 'TreeBackup_n10', 'TreeBackup_n20']
        method_data = {'ExpectedSarsa': {},
                       'TreeBackup_n3': {},
                       'TreeBackup_n5': {},
                       'TreeBackup_n10': {},
                       'TreeBackup_n20': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-1500, -100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='nStep_TreeBackup',
                                  interval_size=50)

        # Decaying Sigma #
        method_names = ['DecayingSigma_OnPolicy', 'DecayingSigma_n3', 'DecayingSigma_n5', 'DecayingSigma_n10',
                        'DecayingSigma_n20']
        method_data = {'DecayingSigma_OnPolicy': {},
                       'DecayingSigma_n3': {},
                       'DecayingSigma_n5': {},
                       'DecayingSigma_n10': {},
                       'DecayingSigma_n20': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-1500, -100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='nStep_DecayingSigma',
                                  interval_size=50)


    ##########################
    """ Best n-Step Methods """
    ##########################
    if args.best_nstep:
        """ Experiment Colors """
        colors = ['#490A3D',  # Purple-ish      - Sarsa n20
                  '#BD1550',  # Red-ish         - QSigma 0.5 n20
                  '#E97F02',  # Orange-ish      - TreeBackup n20
                  '#F8CA00',  # Yellow-ish      - DecayingSigma n20
                  '#8A9B0F',  # Green-ish       - DecayingSigma Hand Picked SD n10
                  '#C0ADDB']  # Violet-ish      - QLearning

        shade_colors = ['#c9bac6',  # Purple-ish        - Sarsa n20
                        '#eab8ca',  # Red-ish           - QSigma 0.5 n20
                        '#f8d8b3',  # Orange-ish        - TreeBackup n20
                        '#f4e8b9',  # Yellow-ish        - DecayingSigma n20
                        '#dde2b8',  # Green-ish         - DecayingSigma HP SD n10
                        "#e8e0f2"   # Violet-ish        - QLearning
                        ]

        method_names = ['Sarsa_n20', 'QSigma0.5_n20', 'TreeBackup_n20', 'DecayingSigma_n20', "DecayingSigma_hp_n20",
                        "QLearning"]
        method_data = {'Sarsa_n20': {},
                       'QSigma0.5_n20': {},
                       'TreeBackup_n20': {},
                       'DecayingSigma_hp_n20': {},
                       'DecayingSigma_n20': {},
                       'QLearning': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.avg_per_episode:
            plot_avg_return_per_episode(method_data, ylim=(-500, -100), ytitle='Average Return per Episode',
                                        xtitle='Episode Number', figure_name='Best_nStep_Methods')
        if args.cumulative_avg:
            plot_cummulative_average(method_data, ylim=(-1000,0), ytitle='Cumulative Average Return', xtitle='Episode Number',
                                     figure_name='Best_nStep_Methods_Cumulative_Avg')
        if args.moving_avg:
            plot_moving_average(method_data, ylim=(-500,-100), ytitle='Cumulative Average Return', xtitle='Episode Number',
                                figure_name='Best_nStep_Methods_Moving_Avg', window=50)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-500,-100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='Best_nStep_Methods_Interval_Avg1',
                                  interval_size=50)

        """ Experiment Colors """
        colors = ['#490A3D',  # Purple-ish      - Sarsa n20
                  '#BD1550',  # Red-ish         - QSigma 0.5 n20
                  '#E97F02',  # Orange-ish      - TreeBackup n20
                  '#F8CA00',  # Yellow-ish      - DecayingSigma n20
                  '#8A9B0F',  # Green-ish       - DecayingSigma Hand Picked SD n10
                  '#C0ADDB']  # Violet-ish      - QLearning

        shade_colors = ['#c9bac6',  # Purple-ish        - Sarsa n20
                        '#eab8ca',  # Red-ish           - QSigma 0.5 n20
                        '#f8d8b3',  # Orange-ish        - TreeBackup n20
                        '#f4e8b9',  # Yellow-ish        - DecayingSigma n20
                        '#dde2b8',  # Green-ish         - DecayingSigma HP SD n10
                        "#e8e0f2"   # Violet-ish        - QLearning
                        ]

        # method_names = ['Sarsa_n10', 'QSigma0.5_n20', 'TreeBackup_n20', 'DecayingSigma_n10_sd0.99723126', "QLearning"]
        # method_data = {'Sarsa_n10': {},
        #                'QSigma0.5_n20': {},
        #                'TreeBackup_n20': {},
        #                'DecayingSigma_n10_sd0.99723126': {},
        #                'QLearning': {}}
        #
        # compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-500,-100),
                                  ytitle='Average Return Over the Last 50 Episodes',
                                  xtitle='Episode Number', figure_name='Best_nStep_Methods_Interval_Avg2',
                                  interval_size=50)

    #################################
    """ Decaying Sigma Comparison """
    #################################
    if args.ds_comparison:

        def experiment_plot(method_data, args, name_suffix):
            if args.avg_per_episode:
                plot_avg_return_per_episode(method_data, ylim=(-600, -100), ytitle='Average Return per Episode',
                                            xtitle='Episode Number',
                                            figure_name='ds_comparison_avg_return_per_episode_' + name_suffix)

            if args.interval_avg:
                plot_interval_average(methods_data=method_data, ylim=(-600, -100),
                                      ytitle='Average Return Over the Last 50 Episodes',
                                      xtitle='Episode Number', figure_name='ds_comparison_interval_avg_' + name_suffix,
                                      interval_size=10, shaded=True)

        """ Experiment Colors """
        colors = ['#E32551',  # Pink      - Original DS
                  '#029DAF',  # Blue      - Modified DS
                  ]

        shade_colors = ['#fbd9e1',  # Light Pink        - Original DS
                        '#d2eef1',  # Light Blue        - Modified DS
                        ]

        ##################################
        """Exponential vs Linear Decay """
        ##################################
        method_names = ['DecayingSigma_n3',
                        'Linearly_DecayingSigma_n3',
                        ]
        method_data = {'DecayingSigma_n3': {},
                       'Linearly_DecayingSigma_n3': {},
                       }

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        experiment_plot(method_data, args, name_suffix='n3')

        method_names = ['DecayingSigma_n5',
                        'Linearly_DecayingSigma_n5',
                        ]
        method_data = {'DecayingSigma_n5': {},
                       'Linearly_DecayingSigma_n5': {},
                       }

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        experiment_plot(method_data, args, name_suffix='n5')

        method_names = ['DecayingSigma_n10',
                        'Linearly_DecayingSigma_n10',
                        ]
        method_data = {'DecayingSigma_n10': {},
                       'Linearly_DecayingSigma_n10': {},
                       }

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        experiment_plot(method_data, args, name_suffix='n10')

        method_names = ['DecayingSigma_n20',
                        'Linearly_DecayingSigma_n20',
                        ]
        method_data = {'DecayingSigma_n20': {},
                       'Linearly_DecayingSigma_n20': {},
                       }

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        experiment_plot(method_data, args, name_suffix='n20')

    #################################
    """ Target Network Comparison """
    #################################
    if args.tnetwork_comparison:
        pass