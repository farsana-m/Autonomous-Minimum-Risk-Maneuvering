
# Autonomous Minimum Risk Maneuvering for Intelligent Vehicles

## Project Overview

This project develops a simulation-based autonomous vehicle emergency response system. When an emergency is triggered, the vehicle transfers control from the driver to an autonomous control system and attempts to perform a Minimum Risk Maneuver (MRM).

The objective is to guide the vehicle towards a safe stopping area while avoiding obstacles and maintaining controlled vehicle motion.

## Key Features

- Manual emergency trigger
- Autonomous control authority transfer simulation
- Proximal Policy Optimization (PPO) based maneuver planning
- Simulated vehicle and obstacle environment
- Safe pull-over and stopping evaluation
- Interactive Streamlit dashboard

## Technologies Used

- Python
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- Gymnasium
- NumPy and Pandas
- Matplotlib
- Streamlit

## Project Structure

- `control/` – Vehicle control modules
- `dashboard/` – Streamlit user interface
- `drl/` – DRL environment, training and evaluation
- `fusion/` – Sensor fusion development
- `models/` – Trained PPO model
- `mrm/` – Minimum Risk Maneuver modules
- `results/` – Evaluation results
- `scenarios/` – Simulation scenarios

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the dashboard:

   `python -m streamlit run dashboard/app.py`

## Current Implementation Scope

This is a software-based simulation prototype. The emergency trigger and vehicle environment are simulated. Physical vehicle hardware and real-world sensor integration are not included in the current prototype.

## Project Status

Under development.
