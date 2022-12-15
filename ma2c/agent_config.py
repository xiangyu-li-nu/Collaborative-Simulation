import rewards
import states
import Evaluation

from agents.madqn import IDQN
from agents.fma2c import FMA2C
from agents.ma2c import MA2C

agent_configs = {
    'IDQN': {
        'agent': IDQN,
        'state': states.drq_norm,
        'reward': rewards.wait,
        'eva': Evaluation.eva,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },

    'MA2C': {
        'agent': MA2C,
        'state': states.ma2c,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'management_acts': 4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 1e-5,
        'max_grad_norm': 40,
        'gamma': 0.96,
        'lr_init': 2.5e-4,
        'lr_decay': 'constant',
        'entropy_coef_init': 0.001,
        'entropy_coef_min': 0.001,
        'entropy_decay': 'constant',
        'entropy_ratio': 0.5,
        'value_coef': 0.5,
        'num_lstm': 64,
        'num_fw': 128,
        'num_ft': 32,
        'num_fp': 64,
        'batch_size': 120,
        'reward_norm': 2000.0,
        'reward_clip': 2.0,
    },

    'FMA2C': {
        'agent': FMA2C,
        'state': states.fma2c,
        'reward': rewards.fma2c,
        'eva': Evaluation.eva,
        'max_distance': 200,
        'management_acts': 4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 1e-5,
        'max_grad_norm': 40,
        'gamma': 0.96,
        'lr_init': 2.5e-4,
        'lr_decay': 'constant',
        'entropy_coef_init': 0.001,
        'entropy_coef_min': 0.001,
        'entropy_decay': 'constant',
        'entropy_ratio': 0.5,
        'value_coef': 0.5,
        'num_lstm': 64,
        'num_fw': 128,
        'num_ft': 32,
        'num_fp': 64,
        'batch_size': 120,
        'reward_norm': 2000.0,
        'reward_clip': 2.0,
    }
}
