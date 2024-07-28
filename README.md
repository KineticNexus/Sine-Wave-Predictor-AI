# Neuroevolution Sine Wave Prediction

This project implements a neuroevolution algorithm to predict future values of a sine wave. It combines simple neural networks with a genetic algorithm to evolve a population of agents that can accurately predict the next value in a sine wave sequence.

## Features

- Generates normalized sine wave data
- Implements a simple neural network as an `Agent` class
- Uses a genetic algorithm for evolving the population of agents
- Provides visualization of predictions vs. actual values
- Tracks and plots average error over generations

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/neuroevolution-sine-prediction.git
   ```
2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```

## Usage

Run the script using Python:

```
python neuroevolution_sine_prediction.py
```

The script will output progress updates during the simulation and display a plot of the results when complete.

## How it Works

1. A population of agents (simple neural networks) is initialized with random weights.
2. In each generation:
   - Agents are evaluated on a subset of the sine wave data.
   - The best-performing agents are selected for reproduction.
   - New agents are created through crossover and mutation.
3. This process continues for a set number of generations or until early stopping criteria are met.
4. The final results are plotted, showing the predictions vs. actual values and the average error over time.

## Parameters

You can adjust the following parameters in the script:

- `num_agents`: Number of agents in the population
- `num_generations`: Maximum number of generations to run
- `input_size`: Number of past data points used for prediction
- `total_days`: Total number of data points in the sine wave

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
