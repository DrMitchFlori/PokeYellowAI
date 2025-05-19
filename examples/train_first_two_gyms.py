"""Example script to train up to the first two gyms using gym-retro."""

import logging
import random

import retro

from rewards import compute_reward


# Map IDs from the disassembly are required to detect milestone completion.
# Replace the placeholders below with the actual values once known.
PEWTER_CITY_MAP_ID = 0  # TODO: insert real map ID for Pewter City
CERULEAN_CITY_MAP_ID = 0  # TODO: insert real map ID for Cerulean City


def main():
    """Runs a basic loop over the ROM and logs when gym milestones are reached."""
    logging.basicConfig(level=logging.INFO)

    env = retro.make(game='PokemonYellow', rom_path='PokemonYellow.gbc')
    obs = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        # Random policy for demonstration purposes.
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        total_reward += compute_reward(info)

        current_map = info.get('map_id')
        if current_map == PEWTER_CITY_MAP_ID:
            logging.info('Reached Pewter City (Gym 1).')
        elif current_map == CERULEAN_CITY_MAP_ID:
            logging.info('Reached Cerulean City (Gym 2).')
            break  # Stop after the second gym

    env.close()
    logging.info('Total reward: %s', total_reward)


if __name__ == '__main__':
    main()
