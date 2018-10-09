import numpy as np
import argparse

from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval, create_results_file, \
    compute_chi2dist_confidence_interval
from Experiments_Engine.Plots_and_Summaries import compute_welchs_test


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-compute_bprobabilities', action='store_true', default=False)

    args = parser.parse_args()

    """
    Results as reported in: Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for 
                            General Agents (2017)
    """
    results = {

        'Alien': {'sarsa_mean': 4272.7, 'sarsa_std': 773.2, 'sarsa_sample_size': 24,
                  'dqn_mean': 2742.0, 'dqn_std': 357.5, 'dqn_sample_size': 5},

        'Amidar': {'sarsa_mean': 411.4, 'sarsa_std': 177.4, 'sarsa_sample_size': 24,
                   'dqn_mean': 792.6, 'dqn_std': 220.4, 'dqn_sample_size': 5},

        'Assault': {'sarsa_mean': 1049.4, 'sarsa_std': 182.7, 'sarsa_sample_size': 24,
                    'dqn_mean': 1424.6, 'dqn_std': 106.8, 'dqn_sample_size': 5},

        'Asterix': {'sarsa_mean': 4358.0, 'sarsa_std': 431.6, 'sarsa_sample_size': 24,
                    'dqn_mean': 2866.8, 'dqn_std': 1354.6, 'dqn_sample_size': 5},

        'Asteroids': {'sarsa_mean': 1524.1, 'sarsa_std': 191.2, 'sarsa_sample_size': 24,
                      'dqn_mean': 528.5, 'dqn_std': 37.0, 'dqn_sample_size': 5},

        'Atlantis': {'sarsa_mean': 38057.5, 'sarsa_std': 8455.2, 'sarsa_sample_size': 24,
                     'dqn_mean': 232442.9, 'dqn_std': 128678.4, 'dqn_sample_size': 5},

        'Bank Heist': {'sarsa_mean': 419.7, 'sarsa_std': 60.5, 'sarsa_sample_size': 24,
                       'dqn_mean': 760.0, 'dqn_std': 82.3, 'dqn_sample_size': 5},

        'Battle Zone': {'sarsa_mean': 25089.6, 'sarsa_std': 4845.9, 'sarsa_sample_size': 24,
                        'dqn_mean': 20547.5, 'dqn_std': 1843.0, 'dqn_sample_size': 5},

        'Beam Rider': {'sarsa_mean': 2234.0, 'sarsa_std': 471.5, 'sarsa_sample_size': 24,
                       'dqn_mean': 5700.5, 'dqn_std': 362.5, 'dqn_sample_size': 5},

        'Berzerk': {'sarsa_mean': 622.3, 'sarsa_std': 70.1, 'sarsa_sample_size': 24,
                    'dqn_mean': 487.2, 'dqn_std': 29.9, 'dqn_sample_size': 5},

        'Bowling': {'sarsa_mean': 64.4, 'sarsa_std': 4.3, 'sarsa_sample_size': 24,
                    'dqn_mean': 33.6, 'dqn_std': 2.7, 'dqn_sample_size': 5},

        'Boxing': {'sarsa_mean': 78.1, 'sarsa_std': 16.5, 'sarsa_sample_size': 24,
                    'dqn_mean': 72.7, 'dqn_std': 4.9, 'dqn_sample_size': 5},

        'Breakout': {'sarsa_mean': 20.2, 'sarsa_std': 1.9, 'sarsa_sample_size': 24,
                     'dqn_mean': 35.1, 'dqn_std': 22.6, 'dqn_sample_size': 5},

        'Carnival': {'sarsa_mean': 3489.8, 'sarsa_std': 2621.5, 'sarsa_sample_size': 24,
                    'dqn_mean': 4803.8, 'dqn_std': 189.0, 'dqn_sample_size': 5},

        'Centipede': {'sarsa_mean': 1189.3, 'sarsa_std': 1040.4, 'sarsa_sample_size': 24,
                      'dqn_mean': 2838.9, 'dqn_std': 225.3, 'dqn_sample_size': 5},

        'Chopper Command': {'sarsa_mean': 2402.8, 'sarsa_std': 806.5, 'sarsa_sample_size': 24,
                            'dqn_mean': 4399.6, 'dqn_std': 401.5, 'dqn_sample_size': 5},

        'Crazy Climber': {'sarsa_mean': 60471.0, 'sarsa_std': 5534.9, 'sarsa_sample_size': 24,
                          'dqn_mean': 78352.1, 'dqn_std': 1967.3, 'dqn_sample_size': 5},

        'Defender': {'sarsa_mean': 10778.6, 'sarsa_std': 1509.0, 'sarsa_sample_size': 24,
                     'dqn_mean': 2941.3, 'dqn_std': 106.2, 'dqn_sample_size': 5},

        'Demon Attack': {'sarsa_mean': 1272.2, 'sarsa_std': 253.6, 'sarsa_sample_size': 24,
                         'dqn_mean': 5182.0, 'dqn_std': 778.0, 'dqn_sample_size': 5},

        'Double Dunk': {'sarsa_mean': -6.9, 'sarsa_std': 0.5, 'sarsa_sample_size': 24,
                        'dqn_mean': -8.7, 'dqn_std': 4.5, 'dqn_sample_size': 5},

        'Elevator Action': {'sarsa_mean': 11147.8, 'sarsa_std': 4291.3, 'sarsa_sample_size': 24,
                            'dqn_mean': 6.0, 'dqn_std': 10.4, 'dqn_sample_size': 5},

        'Enduro': {'sarsa_mean': 294.0, 'sarsa_std': 8.0, 'sarsa_sample_size': 24,
                   'dqn_mean': 688.2, 'dqn_std': 32.4, 'dqn_sample_size': 5},

        'Fishing Derby': {'sarsa_mean': -69.2, 'sarsa_std': 8.9, 'sarsa_sample_size': 24,
                          'dqn_mean': 10.2, 'dqn_std': 1.9, 'dqn_sample_size': 5},

        'Freeway': {'sarsa_mean': 31.8, 'sarsa_std': 0.3, 'sarsa_sample_size': 24,
                    'dqn_mean': 33.0, 'dqn_std': 0.3, 'dqn_sample_size': 5},

        'Frostbite': {'sarsa_mean': 3207.2, 'sarsa_std': 1040.4, 'sarsa_sample_size': 24,
                      'dqn_mean': 279.6, 'dqn_std': 13.9, 'dqn_sample_size': 5},

        'Gopher': {'sarsa_mean': 5555.4, 'sarsa_std': 594.1, 'sarsa_sample_size': 24,
                   'dqn_mean': 3925.5, 'dqn_std': 521.4, 'dqn_sample_size': 5},

        'Gravitar': {'sarsa_mean': 1150.0, 'sarsa_std': 397.5, 'sarsa_sample_size': 24,
                     'dqn_mean': 154.9, 'dqn_std': 17.7, 'dqn_sample_size': 5},

        'HERO': {'sarsa_mean': 14910.2, 'sarsa_std': 3887.6, 'sarsa_sample_size': 24,
                 'dqn_mean': 18843.3, 'dqn_std': 2234.9, 'dqn_sample_size': 5},

        'Ice Hockey': {'sarsa_mean': 12.6, 'sarsa_std': 3.5, 'sarsa_sample_size': 24,
                       'dqn_mean': -3.8, 'dqn_std': 4.7, 'dqn_sample_size': 5},

        'James Bond': {'sarsa_mean': 719.8, 'sarsa_std': 292.0, 'sarsa_sample_size': 24,
                       'dqn_mean': 581.0, 'dqn_std': 21.3, 'dqn_sample_size': 5},

        'Kangaroo': {'sarsa_mean': 4225.8, 'sarsa_std': 2046.9, 'sarsa_sample_size': 24,
                     'dqn_mean': 12291.7, 'dqn_std': 1115.9, 'dqn_sample_size': 5},

        'Krull': {'sarsa_mean': 8894.9, 'sarsa_std': 8482.7, 'sarsa_sample_size': 24,
                  'dqn_mean': 6416.0, 'dqn_std': 128.5, 'dqn_sample_size': 5},

        'Kung Fu Master': {'sarsa_mean': 29915.5, 'sarsa_std': 3647.5, 'sarsa_sample_size': 24,
                           'dqn_mean': 16472.7, 'dqn_std': 2892.7, 'dqn_sample_size': 5},

        'Montezumas Revenge': {'sarsa_mean': 574.2, 'sarsa_std': 590.1, 'sarsa_sample_size': 24,
                               'dqn_mean': 0.0, 'dqn_std': 0.0, 'dqn_sample_size': 5},

        'Ms Pacman': {'sarsa_mean': 4440.5, 'sarsa_std': 616.4, 'sarsa_sample_size': 24,
                      'dqn_mean': 3116.2, 'dqn_std': 141.2, 'dqn_sample_size': 5},

        'Name this Game': {'sarsa_mean': 6750.3, 'sarsa_std': 1376.5, 'sarsa_sample_size': 24,
                           'dqn_mean': 3925.2, 'dqn_std': 660.2, 'dqn_sample_size': 5},

        'Phoenix': {'sarsa_mean': 5197.0, 'sarsa_std': 374.5, 'sarsa_sample_size': 24,
                    'dqn_mean': 2831.0, 'dqn_std': 581.0, 'dqn_sample_size': 5},

        'Pitfall!': {'sarsa_mean': 0.0, 'sarsa_std': 0.0, 'sarsa_sample_size': 24,
                     'dqn_mean': - 21.4, 'dqn_std': 3.2, 'dqn_sample_size': 5},

        'Pong': {'sarsa_mean': 14.5, 'sarsa_std': 2.0, 'sarsa_sample_size': 24,
                 'dqn_mean': 15.1, 'dqn_std': 1.0, 'dqn_sample_size': 5},

        'Pooyan': {'sarsa_mean': 2197.8, 'sarsa_std': 133.7, 'sarsa_sample_size': 24,
                   'dqn_mean': 3700.4, 'dqn_std': 349.5, 'dqn_sample_size': 5},

        'Private Eye': {'sarsa_mean': 44.2, 'sarsa_std': 49.3, 'sarsa_sample_size': 24,
                        'dqn_mean': 3967.5, 'dqn_std': 5540.6, 'dqn_sample_size': 5},

        'Qbert': {'sarsa_mean': 6992.9, 'sarsa_std': 1479.0, 'sarsa_sample_size': 24,
                  'dqn_mean': 9875.5, 'dqn_std': 1385.3, 'dqn_sample_size': 5},

        'River Raid': {'sarsa_mean': 10639.2, 'sarsa_std': 1882.6, 'sarsa_sample_size': 24,
                       'dqn_mean': 10210.4, 'dqn_std': 435.0, 'dqn_sample_size': 5},

        'Road Runner': {'sarsa_mean': 31493.9, 'sarsa_std': 4160.7, 'sarsa_sample_size': 24,
                        'dqn_mean': 42028.3, 'dqn_std': 1492.0, 'dqn_sample_size': 5},

        'Robotank': {'sarsa_mean': 27.3, 'sarsa_std': 2.3, 'sarsa_sample_size': 24,
                     'dqn_mean': 58.0, 'dqn_std': 6.4, 'dqn_sample_size': 5},

        'Seaquest': {'sarsa_mean': 1402.9, 'sarsa_std': 328.3, 'sarsa_sample_size': 24,
                     'dqn_mean': 1485.7, 'dqn_std': 740.8, 'dqn_sample_size': 5},

        'Skiing': {'sarsa_mean': -29940.2, 'sarsa_std': 102.2, 'sarsa_sample_size': 24,
                   'dqn_mean': -12446.6, 'dqn_std': 1257.9, 'dqn_sample_size': 5},

        'Solaris': {'sarsa_mean': 807.3, 'sarsa_std': 216.2, 'sarsa_sample_size': 24,
                    'dqn_mean': 1210.0, 'dqn_std': 148.3, 'dqn_sample_size': 5},

        'Space Invaders': {'sarsa_mean': 759.4, 'sarsa_std': 58.7, 'sarsa_sample_size': 24,
                           'dqn_mean': 823.6, 'dqn_std': 335.0, 'dqn_sample_size': 5},

        'Star Gunner': {'sarsa_mean': 1107.6, 'sarsa_std': 125.6, 'sarsa_sample_size': 24,
                        'dqn_mean': 39269.9, 'dqn_std': 5298.8, 'dqn_sample_size': 5},

        'Tennis': {'sarsa_mean': -0.1, 'sarsa_std': 0.0, 'sarsa_sample_size': 24,
                   'dqn_mean': -23.9, 'dqn_std': 0.0, 'dqn_sample_size': 5},

        'Time Pilot': {'sarsa_mean': 4221.5, 'sarsa_std': 402.1, 'sarsa_sample_size': 24,
                       'dqn_mean': 2061.8, 'dqn_std': 228.8, 'dqn_sample_size': 5},

        'Tutankham': {'sarsa_mean': 91.5, 'sarsa_std': 63.3, 'sarsa_sample_size': 24,
                      'dqn_mean': 60.0, 'dqn_std': 12.7, 'dqn_sample_size': 5},

        'Up and Down': {'sarsa_mean': 15400.1, 'sarsa_std': 14864.6, 'sarsa_sample_size': 24,
                        'dqn_mean': 4750.7, 'dqn_std': 1007.5, 'dqn_sample_size': 5},

        'Venture': {'sarsa_mean': 139.3, 'sarsa_std': 323.2, 'sarsa_sample_size': 24,
                    'dqn_mean': 3.2, 'dqn_std': 4.7, 'dqn_sample_size': 5},

        'Video Pinball': {'sarsa_mean': 13398.0, 'sarsa_std': 3643.7, 'sarsa_sample_size': 24,
                          'dqn_mean': 15398.5, 'dqn_std': 2126.1, 'dqn_sample_size': 5},

        'Wizard of Wor': {'sarsa_mean': 2043.5, 'sarsa_std': 801.3, 'sarsa_sample_size': 24,
                          'dqn_mean': 2231.1, 'dqn_std': 820.8, 'dqn_sample_size': 5},

        'Yars Revenge': {'sarsa_mean': 7257.8, 'sarsa_std': 1884.8, 'sarsa_sample_size': 24,
                         'dqn_mean': 13073.4, 'dqn_std': 1961.8, 'dqn_sample_size': 5},

        'Zaxxon': {'sarsa_mean': 8166.8, 'sarsa_std': 3979.8, 'sarsa_sample_size': 24,
                   'dqn_mean': 3852.1, 'dqn_std': 1120.7, 'dqn_sample_size': 5},

    }

    statistically_significant = 0
    dqn_better = 0
    sarsa_lambda_better = 0
    keys = np.sort(list(results.keys()))
    mock_dqn_better = 0
    mock_sarsa_better = 0

    for key in keys:
        print("-------------------------------------------------")
        print("Game:", key)
        gresults = results[key]
        tvalue, pvalue = compute_welchs_test(gresults['sarsa_mean'], gresults['sarsa_std'],
                                             gresults['sarsa_sample_size'],
                                             gresults['dqn_mean'], gresults['dqn_std'],
                                             gresults['dqn_sample_size'])
        print('The t-value is:', np.round(tvalue, 4))
        print('The p-value is:', np.round(pvalue, 4))

        if tvalue > 0: mock_sarsa_better += 1
        else: mock_dqn_better += 1

        if (pvalue < 0.025) or (pvalue > 0.975):
            print('The difference is statistically significant.')
            statistically_significant += 1

            if tvalue > 0: sarsa_lambda_better += 1
            else: dqn_better += 1

    print("\n---------------------- Statistically Significant Results ---------------------------")
    print('From', len(keys), 'games,', statistically_significant, 'have statistically different performance.')
    print('From the', statistically_significant, 'games with statistically significant performance:')
    print('\tDQN performed better on', dqn_better, 'games.')
    print('\tSarsa(lambda) performed better on', sarsa_lambda_better, 'games.')
    print("------------------------------------------------------------------------------------")

    print("\n--------------------- Mock Results ----------------------------")
    print('DQN performed better on', mock_dqn_better, 'games.')
    print('Sarsa(lambda) performed better on', mock_sarsa_better, 'games.')
    print("---------------------------------------------------------------")