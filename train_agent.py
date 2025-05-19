# coding: utf-8
"""Training script for Pokémon Yellow using Gym Retro and PPO.

This script demonstrates how to train an agent with shaped rewards
based on memory goals.  Goals are loaded from a JSON file and
unlocked gradually using a simple curriculum strategy.
"""

import argparse
import json
import logging
import random

import retro
import torch
import torch.optim as optim
import numpy as np

from ppo import (
    ActorCritic,
    Curriculum,
    gather_rollout,
    load_config,
    ppo_update,
)


logger = logging.getLogger(__name__)



def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train a PPO agent on Pokémon Yellow")
    parser.add_argument(
        "--retro-dir",
        default="integrations",
        help="Path to a Gym Retro integration directory containing Pokemon Yellow",
    )
    parser.add_argument("--goals", default="data/first_three_gyms.json", help="JSON file describing goal curriculum")
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="YAML or JSON file specifying hyperparameters",
    )
    parser.add_argument("--total-steps", type=int, default=100000, help="Total environment steps to train")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Number of steps per PPO rollout")
    parser.add_argument(
        "--output-model",
        default="ppo_pokemon_yellow.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional path to log map IDs during training",
    )
    args = parser.parse_args()

    retro.data.Integrations.add_custom_path(args.retro_dir)
    env = retro.make(game="PokemonYellow-GB")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(args.seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)
    obs_space_shape = env.observation_space.shape
    obs_shape = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1])
    n_actions = env.action_space.n

    config = load_config(args.config)

    with open(args.goals, "r", encoding="utf-8") as f:
        goal_data = json.load(f)
    curriculum = Curriculum(goal_data, threshold=config.get("curriculum", {}).get("threshold", 0.8))

    model = ActorCritic(obs_shape, n_actions)
    ppo_cfg = config.get("ppo", {})
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.get("learning_rate", 2.5e-4))

    clip_range = ppo_cfg.get("clip_range", 0.2)
    epochs = ppo_cfg.get("epochs", 4)
    batch_size = ppo_cfg.get("batch_size", 64)
    vf_coef = ppo_cfg.get("vf_coef", 0.5)
    ent_coef = ppo_cfg.get("ent_coef", 0.01)
    gamma = ppo_cfg.get("gamma", 0.99)
    lam = ppo_cfg.get("lam", 0.95)

    steps = 0
    global_step = 0
    log_f = open(args.log_path, "w") if args.log_path else None
    while steps < args.total_steps:
        rollout = gather_rollout(env, model, curriculum, args.rollout_steps)
        if log_f:
            for mid in rollout.get("map_ids", []):
                log_f.write(f"{global_step}\t{mid}\n")
                global_step += 1
        steps += len(rollout["rewards"])
        ppo_update(
            model,
            optimizer,
            rollout,
            clip_range=clip_range,
            epochs=epochs,
            batch_size=batch_size,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            gamma=gamma,
            lam=lam,
        )
        logger.info(
            "Steps: %s | Active goals: %s",
            steps,
            len(curriculum.active_goals()),
        )

    env.close()
    if log_f:
        log_f.close()
    torch.save(model.state_dict(), args.output_model)


if __name__ == "__main__":
    main()

