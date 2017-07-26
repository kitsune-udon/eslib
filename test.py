import argparse
import pickle
import mlp
import eslib.functions as F
import gym
import numpy as np
from gym.monitoring import VideoRecorder

class Runner:
    def __init__(self, args):
        with open(args.src_filepath, "rb") as f:
            model = pickle.load(f)
        env = gym.make(args.env_name)
        self.model = model
        self.env = env
        self.n_iter = args.n_iter
        self.rec_flag = args.rec
        self.recorder = VideoRecorder(env, base_path=args.src_filepath)

    def __call__(self):
        for i in range(self.n_iter):
            score = self.get_score()
            print(score)

    def close(self):
        self.recorder.close()
        self.env.close()

    def get_score(self):
        env = self.env
        obs = env.reset()
        acc = 0
        while True:
            y = self.model(obs)
            action = np.random.choice(len(y), p=F.softmax(y))
            obs, reward, done, info = env.step(action)
            acc += reward
            if self.rec_flag:
                self.recorder.capture_frame()
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
            help="number of iterations")
    parser.add_argument('--rec', dest='rec', const=True, action='store_const', default=False,
            help="whether to save the video")
    args = parser.parse_args()

    runner = Runner(args)
    try:
        runner()
    finally:
        runner.close()

main()
