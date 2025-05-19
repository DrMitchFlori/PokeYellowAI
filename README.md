# PokeYellowAI

PokeYellowAI is an experimental project aimed at training reinforcement learning agents to play **Pokémon Yellow** using [Gym Retro](https://github.com/openai/retro).  The project will eventually provide scripts for launching the emulator and training agents.

## Requirements

- Python 3.8 or newer
- `gym-retro` (install via `pip install gym-retro`)
- A legally obtained copy of *Pokémon Yellow* (`PokemonYellow.gbc`)

The included ROM is provided only for research convenience.  Make sure you own an original copy of the game before using it.

## Memory Addresses

The following memory addresses have been verified for detecting key milestones in Pokémon Yellow:

- **Current Map ID**: `0xD35E` (1 byte)
- **Badge Flags**: `0xD356` (1 byte)
- **Event Flags** start at `0xD747`

These values come from community documentation of the original game's RAM layout.

## Reward Function Outline

A future reward function can inspect these memory addresses to detect progression:

1. **Map changes** indicate entering new areas.  Reward transitions between significant maps (e.g., towns, routes, gyms) by reading the value at `0xD35E` each frame and checking for changes.
2. **Badge collection** is tracked via the bit field at `0xD356`.  Grant larger rewards whenever a new bit becomes set, meaning a gym has been cleared.
3. **Event flags** contain story milestones.  Check the bytes starting at `0xD747` for newly set bits to reward key events like defeating bosses or receiving key items.

Monitoring these addresses allows the agent to receive sparse, meaningful rewards instead of relying solely on score or progress counters.

## Running

Training and emulator scripts will live in this repository in the future.  Once available, they can be executed similarly to:

```bash
python train.py --rom PokemonYellow.gbc
```

or

```bash
python run_emulator.py --rom PokemonYellow.gbc
```

The exact commands will depend on the final script names and options.  Until then, ensure the required tools are installed so the environment can be created with `retro.make` once configuration files are added.
