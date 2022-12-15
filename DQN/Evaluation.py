import numpy as np

delay = dict()
queue = dict()

def eva(signals, reset=False):

    if reset:
        delay.clear()
        queue.clear()

    for signal_id in signals:

        if len(signals[signal_id].full_observation["num_vehicles"]) != 0:
            for lane in signals[signal_id].lanes:
                if signal_id not in delay.keys():
                    delay[signal_id] = int(signals[signal_id].full_observation[lane]["total_wait"]) / len(signals[signal_id].full_observation["num_vehicles"])
                else:
                    delay[signal_id] += int(signals[signal_id].full_observation[lane]["total_wait"]) / len(signals[signal_id].full_observation["num_vehicles"])
                if signal_id not in queue.keys():
                    queue[signal_id] = int(signals[signal_id].full_observation[lane]["queue"]) / len(signals[signal_id].full_observation["num_vehicles"])
                else:
                    queue[signal_id] += int(signals[signal_id].full_observation[lane]["queue"]) / len(signals[signal_id].full_observation["num_vehicles"])

        else:
            for lane in signals[signal_id].lanes:
                if signal_id not in delay.keys():
                    delay[signal_id] = int(signals[signal_id].full_observation[lane]["total_wait"])
                else:
                    delay[signal_id] += int(signals[signal_id].full_observation[lane]["total_wait"])
                if signal_id not in queue.keys():
                    queue[signal_id] = int(signals[signal_id].full_observation[lane]["queue"])
                else:
                    queue[signal_id] += int(signals[signal_id].full_observation[lane]["queue"])
    return delay, queue


def test_eva(signals, reset=True):

    if reset:
        delay.clear()
        queue.clear()

    for signal_id in signals:

        for lane in signals[signal_id].lanes:
            if signal_id not in delay.keys():
                delay[signal_id] = int(signals[signal_id].full_observation[lane]["total_wait"])
            else:
                delay[signal_id] += int(signals[signal_id].full_observation[lane]["total_wait"])
            if signal_id not in queue.keys():
                queue[signal_id] = int(signals[signal_id].full_observation[lane]["queue"])
            else:
                queue[signal_id] += int(signals[signal_id].full_observation[lane]["queue"])

        # print("ID: ", signal_id, delay[signal_id], queue[signal_id], len(signals[signal_id].full_observation["num_vehicles"]))
        if queue[signal_id] != 0:
            delay[signal_id] = (delay[signal_id]) / len(signals[signal_id].full_observation["num_vehicles"])
            queue[signal_id] = (queue[signal_id])
        else:
            delay[signal_id] = 0
            queue[signal_id] = 0

    return delay, queue