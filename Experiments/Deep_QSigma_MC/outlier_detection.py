import numpy as np
import pickle
import os
import argparse
from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval

SAMPLE_SIZE = 150
NUMBER_OF_EPISODES = 500


if __name__ == "__main__":

    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', action='store', default=None)
    parser.add_argument('-threshold', action='store', default=-4900)
    args = parser.parse_args()

    method_name = args.method
    threshold = args.threshold
    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")

    if method_name is None:
        print("No method name provided.")

    elif method_name == "all":
        method_list = os.listdir(results_path)
        for method in method_list:
            method_results_path = os.path.join(results_path, method)
            if os.path.isdir(method_results_path):
                below_threshold = 0
                for i in range(SAMPLE_SIZE):
                    agent_folder = os.path.join(method_results_path, 'agent_' + str(i+1))
                    agent_results_filepath = os.path.join(agent_folder, 'results.p')
                    with open(agent_results_filepath, mode='rb') as results_file:
                        agent_data = pickle.load(results_file)
                        agent_return_data = agent_data['return_per_episode']
                    average = np.average(agent_return_data)
                    if average <= threshold:
                        # print("\t agent_" + str(i+1) + " is below the threshold. It's average return is:", average)
                        below_threshold += 1
                if below_threshold > 0:
                    print("### Method name", method, "###")
                    print("The number of agents below the threshold is:", below_threshold)

    else:
        method_results_path = os.path.join(results_path, method_name)
        for i in range(SAMPLE_SIZE):
            agent_folder = os.path.join(method_results_path, 'agent_'+ str(i + 1))
            agent_results_filepath = os.path.join(agent_folder, 'results.p')
            with open(agent_results_filepath, mode='rb') as results_file:
                agent_data = pickle.load(results_file)
                agent_return_data = agent_data['return_per_episode']
            print("### Agent", str(i + 1), "###")
            print("The average return is:", np.average(agent_return_data))
            if np.average(agent_return_data) <= threshold:
                for i in range(50, 550, 50):
                    print("The average return after", str(i), "episodes was:", np.average(agent_return_data[0:i]))
