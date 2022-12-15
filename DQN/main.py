
import pathlib
import os
import multiprocessing as mp
import pandas as pd
from multi_signal import MultiSignal
import argparse
from agent_config import agent_configs
from map_config import map_configs
print(55)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default='IDQN')
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--map", type=str, default='cologne3')
    ap.add_argument("--pwd", type=str, default=str(pathlib.Path().absolute()) + os.sep)
    ap.add_argument("--log_dir", type=str, default=str(pathlib.Path().absolute()) + os.sep + 'logs' + os.sep)
    ap.add_argument("--gui", type=bool, default=True)
    ap.add_argument("--libsumo", type=bool, default=True)
    ap.add_argument("--tr", type=int, default=0)
    args = ap.parse_args()

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials + 1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()


def run_trial(args, trial):

    agt_config = agent_configs[args.agent]
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    map_config = map_configs[args.map]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None: route = args.pwd + route
    print('alg.__name__++str(trial): ', alg.__name__ + '-tr' + str(trial))

    # Define the environment
    env = MultiSignal(alg.__name__ + '-tr' + str(trial),
                      args.map,
                      args.pwd + map_config['net'],
                      agt_config['state'],
                      agt_config['reward'],
                      agt_config['eva'],
                      route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                      step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                      max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args.gui,
                      log_dir=args.log_dir, libsumo=args.libsumo, warmup=map_config['warmup'])


    agt_config['episodes'] = int(args.eps * 0.8)
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = args.log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)


    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    agent = alg(agt_config, obs_act, args.map, trial)


    reward = []

    all_wait_time = []
    all_queue = []


    for _ in range(args.eps):
        obs = env.reset()
        done = False
        rewa = 0
        while not done:
            act = agent.act(obs)
            obs, rew, done, info, delay, queue = env.step(act)
            agent.observe(obs, rew, done, info)
            rewa += int((sum(rew.values())) / len(rew))

        D = int(sum(queue.values()))
        W = int(sum(delay.values()))

        all_wait_time.append(W)
        all_queue.append(D)

        reward.append(rewa)

        data = {"Reward": reward, "Waiting time": all_wait_time, "Queue": all_queue}
        pd.DataFrame(data).to_csv("Data recording.csv", encoding='utf-8-sig')

    env.close()


if __name__ == '__main__':
    main()
