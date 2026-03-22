# RIS-Assisted MAC Simulation with Arbitration-Based DRL Framework

This repository contains a simulation framework for a **RIS-assisted wireless MAC system** with an **arbitration-based DRL architecture**.  
The project focuses on modeling the interaction between:

- **MAC channel access decision**
- **RIS mode request**
- **AP-side centralized arbitration**
- **Closed-loop reward feedback**

The goal is to evaluate whether the proposed framework can reduce channel contention and RIS conflicts while improving overall transmission performance.

---

## Project Overview

In this simulation, each UE makes two types of decisions in every time slot:

1. **MAC decision**  
   Decide whether to:
   - listen, or
   - transmit on a selected channel

2. **RIS request decision**  
   Generate a RIS request vector for AP-side RIS mode arbitration

Unlike conventional random access systems, **MAC collision is not immediately determined when a UE selects a channel**.  
Instead, all requests are first collected in **minislot 1**, and the AP performs centralized arbitration in **minislot 2** based on:

- channel selection
- RIS request consistency
- queue status
- delay
- estimated channel quality

---

## Main Features

- Slot-based simulation framework
- Four-minislot procedure
- UE-side dual decision process:
  - MAC action
  - RIS request
- AP-side centralized arbitration
- Support for different transmission outcomes:
  - success
  - MAC contention loss
  - RIS conflict loss
  - transmission failure
  - listen
- Modular structure for future integration of:
  - I-DQN for MAC control
  - DDPG for RIS request generation

---

## Minislot Procedure

Each time slot is divided into four minislots:

### Minislot 1: Request Collection
Each UE:
- observes local state
- selects MAC action
- generates RIS request
- sends request to the AP

No MAC collision is determined at this stage.

### Minislot 2: AP Arbitration
The AP:
- collects all UE requests
- performs RIS arbitration
- performs channel-level MAC arbitration
- decides which UEs are granted or rejected

### Minislot 3: Transmission Execution
Only granted UEs perform data transmission.  
Transmission success is then determined based on the simplified channel/SINR model.

### Minislot 4: Reward and State Update
The environment:
- computes rewards
- updates UE states
- records system statistics

---

## Repository Structure

```bash
.
├── main.py                  # main simulation entry
├── environment.py           # simulation environment
├── ue_agent.py              # UE agent definition
├── ap_arbitrator.py         # AP-side arbitration logic
├── mac_agent.py             # MAC agent (rule-based / DQN placeholder)
├── ris_agent.py             # RIS agent (random / DDPG placeholder)
├── replay_buffer.py         # replay buffer
├── config.py                # simulation parameters
└── README.md
````

> The actual file structure may be adjusted depending on your implementation.

---

## Simulation Outputs

The simulation tracks the following outcomes:

* **Success**
* **MAC contention loss**
* **RIS conflict loss**
* **Transmission failure**
* **Listen**

It can also be extended to evaluate:

* system throughput
* MAC contention ratio
* RIS conflict ratio
* packet success rate
* queueing delay
* convergence behavior of DRL agents

---

## Current Status

This project is currently in the **simulation framework development stage**.

Implemented / planned components:

* [x] basic slot-based environment
* [x] minislot request/arbitration flow
* [x] AP-side arbitration logic
* [x] simplified transmission execution
* [x] reward calculation skeleton
* [ ] I-DQN integration for MAC decision
* [ ] DDPG integration for RIS request generation
* [ ] more realistic channel model
* [ ] baseline comparison experiments

---

## How to Run

Example:

```bash
python main.py
```

If your main file has another name, please replace `main.py` accordingly.

---

## Requirements

Recommended environment:

* Python 3.8+
* NumPy

You can install the basic dependency with:

```bash
pip install numpy
```

If PyTorch is later used for DRL integration:

```bash
pip install torch
```

---

## Research Motivation

This project aims to study a **RIS-assisted MAC protocol** under a distributed decision-making and centralized arbitration framework.
The key idea is to jointly consider:

* MAC channel contention
* RIS mode conflict
* system-level coordination
* closed-loop feedback for learning

This repository serves as the simulation platform for validating the proposed framework.

---

## Future Work

Possible future extensions include:

* integrating full DRL-based MAC control
* integrating DDPG-based RIS request generation
* adding baseline comparisons with traditional MAC schemes
* supporting more realistic wireless channel modeling
* evaluating performance under different UE densities and traffic loads

---

## License

This project is for academic and research purposes.

---

```

---


