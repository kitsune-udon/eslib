import argparse
import pickle
import mlp
import eslib.functions as F
import gym
import numpy as np

def get_score(model, env):
    obs = env.reset()
    acc = 0
    while True:
        y = model(obs)
        action = np.random.choice(len(y), p=F.softmax(y))
        obs, reward, done, info = env.step(action)
        acc += reward
        if done:
            break
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str, required=True,
            help="gym env name")
    parser.add_argument('--src', dest='src_filepath', type=str, required=True,
            help="input file name")
    parser.add_argument('--n_iter', dest='n_iter', type=int, default=10,
            help="input file name")
    args = parser.parse_args()

    with open(args.src_filepath, "rb") as f:
        model = pickle.load(f)

    env = gym.make(args.env_name)
    for i in range(args.n_iter):
        score = get_score(model, env)
        print(score)
    env.close()


main()
