import eslib
import eslib.links as L
import eslib.functions as F
import eslib.optimizers as O
from eslib.optimizer import GradientClipping, WeightDecay
import numpy as np
import gym

class MLP(eslib.Chain):
    def __init__(self, n_actions):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 10)
            self.l2 = L.Linear(None, n_actions)
    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return h

def get_score(model, env):
    obs = env.reset()
    acc = 0
    while True:
        y = model(obs)
        act = np.random.choice(len(y), p=F.softmax(y))
        obs, reward, done, info = env.step(act)
        acc += reward
        if done:
            break
    total_score = acc / 200
    return total_score


def main():
    n_iters = 1001
    n_ptbs = 16
    env_name = 'CartPole-v0'
    envs = [gym.make(env_name) for _ in range(n_ptbs)]
    envs[0] = gym.wrappers.Monitor(envs[0], env_name + '_monitor', force=True)
    n_actions = envs[0].action_space.n
    p = eslib.Perturbation()
    model = MLP(n_actions)
    optimzier = O.SMORMS3(lr=1e-2)
    optimzier.setup(model)
    optimzier.add_hook(GradientClipping(1e2))
    optimzier.add_hook(WeightDecay(0.005))
    eslib.fix_model(model, gym.make(env_name).reset())

    for i in range(n_iters):
        scores = []
        with eslib.perturbation_scope():
            for ptb_id in range(n_ptbs):
                eslib.set_perturbations(model, p, ptb_id)
                score = get_score(model, envs[ptb_id])
                scores.append(score)
        print("(iter, score ave):({}, {:0.2f})".format(i, np.array(scores[0]).mean()))
        eslib.calculate_grads(model, p, scores)
        optimzier.update()
        model.cleargrads()
        p.age()

main()
