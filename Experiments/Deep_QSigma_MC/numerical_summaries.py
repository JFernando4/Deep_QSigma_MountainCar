import os
import pickle
import numpy as np
import argparse

from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval, create_results_file


def get_agents_data(results_path, method_name, sample_size=100):
    """ Returns the data for all the agents corresponding to one method """
    method_dir = os.path.join(results_path, method_name)
    return_data = []
    for i in range(sample_size):
        agent_path = os.path.join(method_dir, 'agent_' + str(i+1))
        agent_data_path = os.path.join(agent_path, 'results.p')
        with open(agent_data_path, mode='rb') as agent_data_file:
            agent_data = pickle.load(agent_data_file)
        return_data.append(agent_data['return_per_episode'])
    return return_data


def sample_agents(max_agents, num_agents, agents_dir_list):
    agents_idxs = []
    while len(agents_idxs) != max_agents:
        idx = np.random.choice(num_agents)
        if ("agent" in agents_dir_list[idx].split("_")) and (idx not in agents_idxs):
            agents_idxs.append(idx)
    return agents_idxs


def results_summary_data(pathname, evaluation_episodes, average_window, omit_list=[], ci_error=0.05,
                         name="", fa_windows=[50,500], drop_k=0, max_sample_size=100, use_ste=False):
    addtofile = False
    for dir_name in os.listdir(pathname):
        if dir_name in omit_list:
            pass
        else:
            dir_path = os.path.join(pathname, dir_name)
            print("Working on", dir_name + str("..."))

            moving_averages = []
            episode_interval_averages = []
            if os.path.isdir(dir_path):
                agents_list = os.listdir(dir_path)
                top_agents_names = drop_bottom_k(agents_list, drop_k, method_path=dir_path,
                                                 max_sample_size=max_sample_size)

                for i in top_agents_names:
                    agent_path = os.path.join(dir_path, 'agent_'+str(i))
                    agent_data_path = os.path.join(agent_path, 'results.p')
                    with open(agent_data_path, mode='rb') as agent_data_file:
                        agent_data = pickle.load(agent_data_file)
                    agents_results1 = get_data_for_moving_average(agent_data, evaluation_episodes, average_window)
                    moving_averages.append(agents_results1)

                    agent_results2 = get_data_for_episode_interval_average(agent_data, fa_windows)
                    episode_interval_averages.append(agent_results2)

                " Moving Averages txt File "
                moving_averages = np.array(moving_averages)
                sample_size = len(top_agents_names)
                # assert sample_size + drop_k == total_agents
                average = np.average(moving_averages, axis=0)
                std = np.std(moving_averages, axis=0, ddof=1)
                upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(average, std, ci_error,
                                                                                           sample_size)
                method_name = [dir_name] * len(evaluation_episodes)
                deviation_header = 'Standard Deviation' if not use_ste else 'Standard Error'
                headers = ["Method Name", "Evaluation Episode", "Average", deviation_header, "Lower C.I. Bound",
                           "Upper C.I. Bound", "Margin of Error"]
                deviation = np.round(std, 2) if not use_ste else np.round(std / np.sqrt(sample_size), 2)
                columns = [method_name, evaluation_episodes, np.round(average, 2), deviation,
                           np.round(lower_bound, 2), np.round(upper_bound, 2), np.round(error_margin, 2)]
                title = dir_name + " (sample_size = " + str(sample_size) + ")"
                create_results_file(pathname, headers=headers, columns=columns, addtofile=addtofile,
                                    results_name="results_moving_averages_"+name, title=title)

                " Episode Interval Averages txt File "
                episode_interval_averages = np.array(episode_interval_averages)
                average = np.average(episode_interval_averages, axis=0)
                std = np.std(episode_interval_averages, axis=0, ddof=1)
                upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(average, std, ci_error,
                                                                                           sample_size)
                method_name = [dir_name] * len(fa_windows)
                headers = ["Method Name", "Episode Interval", "Average", deviation_header, "Lower C.I. Bound",
                           "Upper C.I. Bound", "Margin of Error"]
                deviation = np.round(std, 2) if not use_ste else np.round(std / np.sqrt(sample_size), 2)
                columns = [method_name, fa_windows, np.round(average, 2), deviation,
                           np.round(lower_bound, 2), np.round(upper_bound, 2), np.round(error_margin, 2)]
                create_results_file(pathname, headers=headers, columns=columns, addtofile=addtofile,
                                    results_name="results_episode_interval_averages_"+name, title=title)
                addtofile = True
    return

# Moving average table
def get_data_for_frame_interval_average(agent_data, evaluation_frames, average_window):
    env_info = agent_data["env_info"]
    if not isinstance(env_info, np.ndarray):
        env_info = np.array(env_info)
    num_frames = len(evaluation_frames)
    averages = np.zeros(num_frames)
    for i in range(num_frames):
        idx = np.argmax(env_info >= evaluation_frames[i])
        averages[i] = np.average(agent_data["return_per_episode"][(idx - average_window): idx])
    return averages


# Moving Average Table
def get_data_for_moving_average(agent_data, evaluation_episodes, average_window):
    num_evaluation_points = len(evaluation_episodes)
    averages = np.zeros(num_evaluation_points, dtype=np.float64)
    idx = 0
    for evaluation_point in evaluation_episodes:
        averages[idx] = np.average(agent_data['return_per_episode'][evaluation_point-average_window:evaluation_point])
        idx += 1
    return averages


# Episode Interval Averages
def get_data_for_episode_interval_average(agent_data, average_windows):
    averages = np.zeros(len(average_windows))
    idx = 0
    for aw in average_windows:
        averages[idx] = np.average(agent_data['return_per_episode'][0:aw])
        idx += 1
    return averages


# Drops the bottom k agents
def drop_bottom_k(agents_list, drop_k, method_path, max_sample_size=100):
    averages = []
    names = []
    sample_counter = 0
    for i in range(len(agents_list)):
        agent_folder = os.path.join(method_path, 'agent_' + str(i + 1))
        agent_results_filepath = os.path.join(agent_folder, 'results.p')
        with open(agent_results_filepath, mode='rb') as results_file:
            agent_data = pickle.load(results_file)
            agent_return_data = agent_data['return_per_episode']
        averages.append(np.average(agent_return_data))
        names.append(i+1)
        sample_counter += 1
        if sample_counter >= max_sample_size:
            break
    names = np.array(names, dtype=np.int32)
    return names[np.argsort(averages)[drop_k:]]


if __name__ == "__main__":

    """ Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('-drop_k', action='store', default=0, type=int)
    parser.add_argument('-threshold', action='store', default=-4900)
    parser.add_argument('-max_sample_size', action='store', default=100, type=int)
    parser.add_argument('-avg_window', action='store', default=50, type=int)
    parser.add_argument('-use_ste', action='store_true', default=False)
    parser.add_argument('-numerical_summaries', action='store_true', default=False)
    parser.add_argument('-ds_comparison', action='store_true', default=False)
    args = parser.parse_args()

    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")

    if args.numerical_summaries:
        fa_windows = [10, 50, 100, 250, 500]
        # evaluation_episodes = [50,100,150,200,250,300,350,400,450,500]
        average_window = args.avg_window
        evaluation_episodes = np.arange(average_window, 500 + average_window, average_window)
        omit_list = []
        results_summary_data(results_path, evaluation_episodes, average_window, ci_error=0.05,
                             name="dropped_"+str(args.drop_k), fa_windows=fa_windows, omit_list=omit_list,
                             drop_k=args.drop_k, max_sample_size=args.max_sample_size, use_ste=args.use_ste)

    if args.ds_comparison:

        methods = [
            {
                'name': 'DecayingSigma_n3',
                'interval': (120, 190)    # avg window = 10
                # 'interval': (140, 200)      # avg window = 20
            },
            {
                'name': 'Linearly_DecayingSigma_n3',
                'interval': (160, 230)    # avg window = 10
                # 'interval': (160, 250)      # avg window = 20
            },
        ]

        def retrieve_data_for_ds_comparison(results_path, method, sample_size=None, average_window=10):
            return_data = np.array(get_agents_data(results_path, method['name'], sample_size), dtype=np.float64)

            interval_start = method['interval'][0]
            interval_end = method['interval'][1]
            first_average = np.average(return_data[:, interval_start-average_window:interval_start], axis=1)
            last_average = np.average(return_data[:, interval_end-average_window:interval_end], axis=1)

            performance_difference = first_average - last_average
            mean_diff = np.average(performance_difference)
            std_diff = np.std(performance_difference, ddof=1)
            proportion = 0.05
            sample_size = performance_difference.size if sample_size is None else sample_size
            upper_ci, lower_ci, margin = compute_tdist_confidence_interval(mean_diff, std_diff, proportion, sample_size)

            print('--------------------------------------------------------')
            print('Method Name:', method['name'])
            print('Interval:', method['interval'])
            print('Average Drop in Performance:', np.round(mean_diff, 2))
            print('Sample Standard Deviation', np.round(std_diff,2))
            print('Standard Error:', np.round(std_diff / np.sqrt(sample_size), 2))
            print('Confidence Interval (Lower, Upper):', (np.round(lower_ci, 2), np.round(upper_ci, 2)))
            print('--------------------------------------------------------')

        avg_window = args.avg_window
        retrieve_data_for_ds_comparison(results_path, methods[0], sample_size=1000, average_window=avg_window)
        retrieve_data_for_ds_comparison(results_path, methods[1], sample_size=1000, average_window=avg_window)
