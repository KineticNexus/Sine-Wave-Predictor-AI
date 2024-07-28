import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
    
    def predict(self, inputs):
        return np.dot(inputs, self.weights)[0]
    
    def mutate(self, rate):
        self.weights += np.random.randn(*self.weights.shape) * rate

def generate_sine_data(days):
    data = np.sin(np.linspace(0, 8*np.pi, days))  # Increased frequency
    return (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0, 1]

def evaluate_agents(agents, sine_data, start_day):
    inputs = sine_data[start_day:start_day+100]  # Reduced input size
    target = sine_data[start_day+100]
    predictions = np.array([agent.predict(inputs) for agent in agents])
    errors = np.abs(predictions - target)
    return errors, predictions

def select_and_reproduce(agents, errors, num_survivors):
    survivor_indices = np.argsort(errors)[:num_survivors]
    survivors = [agents[i] for i in survivor_indices]
    new_agents = []
    for _ in range(len(agents)):
        parent1, parent2 = np.random.choice(survivors, 2, replace=False)
        child = Agent(100, 1)  # Adjusted input size
        # Improved crossover
        crossover_point = np.random.randint(0, 100)
        child.weights[:crossover_point] = parent1.weights[:crossover_point]
        child.weights[crossover_point:] = parent2.weights[crossover_point:]
        child.mutate(rate=0.01)  # Reduced mutation rate
        new_agents.append(child)
    return new_agents

# Initialize parameters
num_agents = 1000
num_generations = 1000  # Reduced number of generations
input_size = 100  # Reduced input size
total_days = 5000  # Reduced total days
sine_data = generate_sine_data(total_days)

# Initialize agents
agents = [Agent(input_size, 1) for _ in range(num_agents)]

predictions = []
actual_values = []
days = []
avg_errors = []

print("Starting simulation...")
for generation in range(num_generations):
    start_day = np.random.randint(0, total_days - input_size - 1)
    errors, gen_predictions = evaluate_agents(agents, sine_data, start_day)
    
    average_prediction = np.mean(gen_predictions)
    actual_value = sine_data[start_day+input_size]
    
    predictions.append(average_prediction)
    actual_values.append(actual_value)
    days.append(start_day + input_size)
    avg_errors.append(np.mean(errors))
    
    agents = select_and_reproduce(agents, errors, num_agents // 4)  # Increased selection pressure
    
    if generation % 50 == 0:
        print(f"Generation {generation}: Avg Error = {avg_errors[-1]:.4f}, Day = {days[-1]}")
    
    # Early stopping
    if generation > 100 and np.mean(avg_errors[-10:]) > np.mean(avg_errors[-20:-10]):
        print(f"Early stopping at generation {generation}")
        break

print(f"Simulation complete. Total generations: {len(predictions)}")

# Sorting the results by day for proper plotting
sorted_indices = np.argsort(days)
sorted_days = np.array(days)[sorted_indices]
sorted_predictions = np.array(predictions)[sorted_indices]
sorted_actual_values = np.array(actual_values)[sorted_indices]

# Plotting
plt.figure(figsize=(15, 12))

plt.subplot(2, 1, 1)
plt.scatter(sorted_days, sorted_predictions, label='Average Prediction', alpha=0.7, s=10)
plt.scatter(sorted_days, sorted_actual_values, c='r', label='Actual Value', alpha=0.7, s=10)
plt.plot(range(total_days), sine_data, 'g-', label='Full Sine Wave', alpha=0.3)
plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Neuroevolution: Sine Wave Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(range(len(avg_errors)), avg_errors)
plt.xlabel('Generation')
plt.ylabel('Average Error')
plt.title('Average Error over Generations')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Plot displayed. Close the plot window to end the program.")