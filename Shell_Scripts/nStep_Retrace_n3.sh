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
    python3 ./Experiments/Deep_nStep_Retrace/deep_nstep_retrace.py -episodes 500 -n 3 -quiet -dump_agent \
    -name Retrace_n3/agent_$i
done
