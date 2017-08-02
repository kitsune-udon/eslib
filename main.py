import eslib
import eslib.links as L
import eslib.functions as F
import eslib.optimizers as O
from eslib.optimizer import GradientClipping, WeightDecay
import numpy as np
from mpi4py import MPI
import pickle, os, gym, mlp, argparse, glob

class Runner:
    def __init__(self, args):
        self.args = args

        comm = MPI.COMM_WORLD
        self.comm = comm
        self.n_workers = comm.size
        self.rank = comm.rank

        if args.clean:
            self.clean_model_dir()

        env = gym.make(args.env_name)
        if args.monitor and self.rank == 0:
            env = gym.wrappers.Monitor(env, args.env_name + '_monitor', force=True)
        self.env = env

        self.n_ptbs = self.n_workers

        self.p = eslib.Perturbation()
        n_actions = env.action_space.n
        self.model = mlp.MLP(n_actions)

        optimizer = O.Adam(alpha=1e-2)
        optimizer.setup(self.model)
        optimizer.add_hook(GradientClipping(1e2))
        optimizer.add_hook(WeightDecay(0.0005))
        self.optimizer = optimizer

        eslib.fix_model(self.model, gym.make(args.env_name).reset())

    def __call__(self):
        for i in range(self.args.n_iter):
            d = np.zeros(2, dtype=np.float32)
            ds = np.zeros(2 * self.n_ptbs, dtype=np.float32)

            with eslib.perturbation_scope():
                ptb_id = self.rank
                eslib.set_perturbations(self.model, self.p, ptb_id)
                score, n_steps = self.get_score()
                d[0], d[1] = score, n_steps

            self.comm.Allgather([d, MPI.FLOAT], [ds, MPI.FLOAT])

            ds = ds.reshape(-1, 2).transpose()
            scores, steps = ds[0], ds[1]

            fitness = eslib.fitness_shaping(scores, self.args.cutoff_fitness) if self.args.fitness_shaping else scores

            eslib.calculate_grads(self.model, self.p, fitness)
            self.optimizer.update()

            if self.rank == 0:
                self.print_status(i, scores, steps)

                if i % self.args.save_interval == 0:
                    self.save_model(i)

            self.model.cleargrads()
            self.p.age()

    def close(self):
        self.env.close()

    @property
    def model_dir(self):
        if self.args.dst_filepath is None:
            dirpath = self.args.env_name + "_model"
        else:
            dirpath = self.args.dst_filepath
        return dirpath

    def save_model(self, i):
        dirpath = self.model_dir

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        filename = "{:0>5}.pickle".format(i)
        filepath = '/'.join([dirpath, filename])

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        print("saved to {}".format(filepath))

    def clean_model_dir(self):
        dirpath = self.model_dir
        if not os.path.exists(dirpath):
            return
        target = []
        target += glob.glob(dirpath + "/*.pickle")
        target += glob.glob(dirpath + "/*.pickle.mp4")
        target += glob.glob(dirpath + "/*.pickle.meta.json")
        for filepath in target:
            os.remove(filepath)

    def print_status(self, i, scores, steps):
        print("iter:{}".format(i))
        print("score(max,ave,min):({:.2f}, {:.2f}, {:.2f})".format(scores.max(), scores.mean(), scores.min()))
        print("step(max,ave,min):({:.2f}, {:.2f}, {:.2f})".format(steps.max(), steps.mean(), steps.min()))
        print("param(max,min):({:.2f}, {:.2f})".format(
            max([param.data.max() for param in self.model.params(False)]),
            min([param.data.min() for param in self.model.params(False)])))
        print("grad(max,min):({:.2f}, {:.2f})".format(
            max([param.grad.max() for param in self.model.params(False)]),
            min([param.grad.min() for param in self.model.params(False)])))

    def get_score(self):
        env = self.env
        obs = env.reset()
        acc = 0
        n_steps = 0
        while True:
            y = self.model(obs)
            action = np.random.choice(len(y), p=F.softmax(y))
            obs, reward, done, info = env.step(action)
            acc += reward
            n_steps += 1
            if done:
                break
        total_score = acc / 200
        return total_score, n_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str, required=True,
            help="gym env name")
    parser.add_argument('--dst', dest='dst_filepath', type=str,
            help="path for saving the model")
    parser.add_argument('--n_iter', dest='n_iter', type=int, default=10001,
            help="number of iterations")
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=10,
            help="interval of saving the model")
    parser.add_argument('--monitor', dest='monitor', const=True, action='store_const', default=False,
            help="whether to monitor the env")
    parser.add_argument('--clean', dest='clean', const=True, action='store_const', default=False,
            help="whether to clean the directory for saving the model")
    parser.add_argument('--fitness_shaping', dest='fitness_shaping', const=True, action='store_const', default=False,
            help="whether to use 'fitness shaping'")
    parser.add_argument('--cutoff_fitness', dest='cutoff_fitness', const=True, action='store_const', default=False,
            help="whether to cutoff the fitness in the fitness shaping")
    args = parser.parse_args()

    runner = Runner(args)
    try:
        runner()
    finally:
        runner.close()

main()
