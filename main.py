import eslib
import eslib.links as L
import eslib.functions as F
import eslib.optimizers as O
from eslib.optimizer import GradientClipping, WeightDecay
import numpy as np
from mpi4py import MPI
import pickle, os, gym, mlp

def save_model(model, env_name, i):
    dirpath = "/tmp/" + env_name
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filename = "{:0>5}.pickle".format(i)
    filepath = '/'.join([dirpath, filename])
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("saved to {}".format(filepath))

def print_status(model, i, scores, steps):
    print("iter:{}".format(i))
    print("score(max,ave,min):({:.2f}, {:.2f}, {:.2f})".format(scores.max(), scores.mean(), scores.min()))
    print("step(max,ave,min):({:.2f}, {:.2f}, {:.2f})".format(steps.max(), steps.mean(), steps.min()))
    print("param(max,min):({:.2f}, {:.2f})".format(
        max([param.data.max() for param in model.params(False)]),
        min([param.data.min() for param in model.params(False)])))

def get_score(model, env):
    obs = env.reset()
    acc = 0
    n_steps = 0
    while True:
        y = model(obs)
        act = np.random.choice(len(y), p=F.softmax(y))
        obs, reward, done, info = env.step(act)
        acc += reward
        n_steps += 1
        if done:
            break
    total_score = acc / 200
    return total_score, n_steps

def main():
    n_iters = 10001
    comm = MPI.COMM_WORLD
    n_workers = comm.size
    rank = comm.rank
    n_ptbs = n_workers
    saving_span = 10

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    if rank == 0:
        env = gym.wrappers.Monitor(env, env_name + '_monitor', force=True)
    n_actions = env.action_space.n

    p = eslib.Perturbation()
    model = mlp.MLP(20, n_actions)
    optimzier = O.SMORMS3(lr=1e-2)
    optimzier.setup(model)
    optimzier.add_hook(GradientClipping(1e2))
    optimzier.add_hook(WeightDecay(0.005))
    eslib.fix_model(model, gym.make(env_name).reset())

    for i in range(n_iters):
        d = np.zeros(2, dtype=np.float32)
        ds = np.zeros(2 * n_ptbs, dtype=np.float32)

        with eslib.perturbation_scope():
            ptb_id = rank
            eslib.set_perturbations(model, p, ptb_id)
            score, n_steps = get_score(model, env)
            d[0], d[1] = score, n_steps

        comm.Allgather([d, MPI.FLOAT], [ds, MPI.FLOAT])

        ds = ds.reshape(-1, 2).transpose()
        scores, steps = ds[0], ds[1]

        eslib.calculate_grads(model, p, scores)
        optimzier.update()
        model.cleargrads()
        p.age()

        if rank == 0:
            print_status(model, i, scores, steps)

            if i % saving_span == 0:
                save_model(model, env_name, i)

main()
