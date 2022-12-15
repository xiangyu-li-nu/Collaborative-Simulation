import torch


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError


class IndependentAgent(Agent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__()
        self.config = config
        self.agents = dict()

    def act(self, observation):
        acts = dict()
        for agent_id in observation.keys():
            acts[agent_id] = self.agents[agent_id].act(observation[agent_id])
        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.agents[agent_id].observe(observation[agent_id], reward[agent_id], done, info)
            if done:
                if info['eps'] % 1 == 0:
                    self.agents[agent_id].save(self.config['log_dir']+'agent_'+agent_id)


    def load_model(self, observation):

        for agent_id in observation.keys():
            # print("posiiton: ", self.config['log_dir'] + 'agent_' + agent_id)
            model = torch.load("SUMO/test_model/"+'agent_'+agent_id+'.pt',  map_location='cpu')
            self.agents[agent_id].load(model)
        print('Load model data')