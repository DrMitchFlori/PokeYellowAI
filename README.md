# PokeYellowAI

Utilities for training reinforcement learning agents on **Pokémon Yellow**.

This repository ships a ROM (`PokemonYellow.gbc`) and helper Python code for
connecting to a Gym Retro compatible emulator.

## Reward Helpers

The `reward.py` module exposes functions to:

* Connect to a Game Boy emulator via Gym Retro.
* Read bytes from important WRAM addresses.
* Evaluate training goals defined in a JSON specification.

A sample goal file is provided in `first_two_gyms.json`.

### Example

```python
import reward

# Create the Retro environment. The game must be registered with Gym Retro.
env = reward.connect_emulator("PokemonYellow-GB")

# Load goal predicates
spec = reward.load_goal_spec("first_two_gyms.json")

observation = env.reset()
for step in range(1000):
    # Replace with your own policy or training algorithm
    action = env.action_space.sample()
    observation, _reward, done, _info = env.step(action)

    ram = reward.read_wram_bytes(env)
    status = reward.evaluate_goals(ram, spec)
    if status[0]["met"]:
        print("First gym reached!")
        break

    if done:
        observation = env.reset()
```
