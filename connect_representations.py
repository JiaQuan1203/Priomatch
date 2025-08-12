import sys
sys.path.append('.')

import argparse
import os

import numpy as np
import pandas as pd


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset for K-Means evaluation", required=True)
    parser.add_argument('--phis', type=str, default="clipvitL14", nargs='+', help="Representation spaces to run K-Means")
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    return parser.parse_args(args)


def run(args=None):
    args = _parse_args(args)

    Zs_train = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_train.npy").astype(np.float32) for phi in args.phis]
    Zs_val = [np.load(f"{args.root_dir}/representations/{phi}/{args.dataset}_val.npy").astype(np.float32) for phi in args.phis]
    Zs_train = [Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True) for Z_train in Zs_train]
    Zs_val = [Z_val / np.linalg.norm(Z_val, axis=1, keepdims=True) for Z_val in Zs_val]
    Ztrain = np.concatenate(Zs_train, axis=1)
    Zval = np.concatenate(Zs_val, axis=1)

    if not os.path.exists(f"{args.root_dir}/representations/{args.phis[0]}_{args.phis[1]}"):
        os.makedirs(f"{args.root_dir}/representations/{args.phis[0]}_{args.phis[1]}")

    np.save(f"{args.root_dir}/representations/{args.phis[0]}_{args.phis[1]}/{args.dataset}_train.npy", Ztrain)
    np.save(f"{args.root_dir}/representations/{args.phis[0]}_{args.phis[1]}/{args.dataset}_val.npy", Zval)

if __name__ == '__main__':
    run()