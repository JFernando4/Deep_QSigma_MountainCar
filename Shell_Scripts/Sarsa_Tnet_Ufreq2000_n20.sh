#!/bin/sh

echo Enter the number of agents you want to train:
read number_of_agents
echo Enter the number of the first agent:
read first_agent_number
last_agent_number=$(($first_agent_number + $number_of_agents - 1))

export PYTHONPATH=.
for (( i=$first_agent_number; i <= $last_agent_number; i++ ))
do
    echo "Training Agent $i"
    python3 ./Experiments/Deep_QSigma_MC/dqsigman_mc.py -episodes 500 -n 20 -sigma 1 -beta 1 \
    -quiet -dump_agent -compute_bprobabilities -target_epsilon 0.1 -tnetwork_update_freq 2000 \
    -name Sarsa_Tnet_Ufreq2000_n20/agent_$i
done
