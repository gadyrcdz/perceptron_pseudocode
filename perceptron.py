import math

##Data 
degrees_fahrenheit =  [
    -20.0000, -8.4211, 3.1579, 14.7368, 26.3158,
    37.8947, 49.4737, 61.0526, 72.6316, 84.2105,
    95.7895, 107.3684, 118.9474, 130.5263, 142.1053,
    153.6842, 165.2632, 176.8421, 188.4211, 200.0000
]

degrees_celsius = [(f - 32) * 5 / 9 for f in degrees_fahrenheit]
# Implement the pseudocode


### fahrenheit_to_celsius(f)
def fahrenheit_to_celsius(f):
    return (f-32)*5/9

### mean

def mean(elements):
    if not elements:
        raise ValueError("Cannot compute standard deviation of an empty list")


    total = 0

    for i in elements:
        total += i
    
    return total/len(elements)


### standard_deviation(elements)

def standard_deviation(elements):
    if not elements:
        raise ValueError("Cannot compute standard deviation of an empty list")

    
    mean_value = mean(elements)
    variance = sum((x - mean_value) ** 2 for x in elements) / len(elements)
    return math.sqrt(variance)


### normalize(dataset)

def normalize(dataset):
    
    value_hat = mean(dataset)
    value_sd = standard_deviation(dataset)

    if value_sd == 0:
        raise ValueError("Cannot normalize values with zero standard deviation")
    
    value_norm = []

    for i in dataset:
        value = (i - value_hat)/ value_sd
        value_norm.append(value)

    return value_norm


### denormalize(value, dataset)

def denormalize(value, dataset):

    value_hat = mean(dataset)
    value_sd = standard_deviation(dataset)


    return value*value_sd + value_hat


### Normalized Datasets

x_normalized = normalize(degrees_fahrenheit)
y_normalized = normalize(degrees_celsius)


### perceptron(x, w, b)

def perceptron(x, w, b):
    return w*x+b

### train_perceptron(x_values, y_values, w, b, learning_rate, epochs)
def train_perceptron(x_values, y_values, w, b, learning_rate, epochs):
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if not x_values:
        raise ValueError("Cannot train with an empty dataset")

    n = len(x_values)

    training_history = {
        "epochs":  [],
        "weights": [],
        "biases":  [],
        "mse":     []
    }

    for epoch in range(epochs):
        dw  = 0
        db  = 0
        mse = 0

        for x, y in zip(x_values, y_values):
            y_hat = perceptron(x, w, b)
            error = y - y_hat
            mse  += error ** 2
            dw   += x * error
            db   += error

        mse = mse / n

        d_mse_dw = (-2 / n) * dw
        d_mse_db = (-2 / n) * db

        w -= learning_rate * d_mse_dw
        b -= learning_rate * d_mse_db

        training_history["epochs"].append(epoch + 1)
        training_history["weights"].append(w)
        training_history["biases"].append(b)
        training_history["mse"].append(mse)

        print(f"Epoch {epoch + 1}  w={w:.6f}  b={b:.6f}  mse={mse:.6f}")

    return w, b, training_history


### Train and Predict


if __name__ == "__main__":
    w = 0
    b = 0
    learning_rate = 0.3
    epochs = 8

    print(f"Mean x_normalized:  {mean(x_normalized)}")
    print(f"SD   x_normalized:  {standard_deviation(x_normalized)}\n")

    w, b, training_history = train_perceptron(
        x_normalized,
        y_normalized,
        w, b,
        learning_rate,
        epochs
    )

    print(f"\nFinal weight: {w}")
    print(f"Final bias:   {b}")

    fahrenheit_value = 5
    fahrenheit_normalized = (fahrenheit_value - mean(degrees_fahrenheit)) / standard_deviation(degrees_fahrenheit)

    predicted_celsius_normalized = perceptron(fahrenheit_normalized, 1.0000000000000184, 5.53443326385718e-17)
    predicted_celsius = denormalize(predicted_celsius_normalized, degrees_celsius)
    # print("Dylan: " + str(denormalize(113.58555406908977, degrees_celsius)))



    print(f"\nPredicted normalized Celsius: {predicted_celsius_normalized}")
    print(f"Predicted Celsius:{predicted_celsius:.4f}")




##Questions to answer:

# What happens to the MSE when you train for more epochs? MSE turns 0
# What happens if the learning rate is too small? The agent make small changes
# What happens if the learning rate is too large? Make biggers changes
# Why do we normalize before training? Without normalize the numbers will turn bigger and bigger
# What do w and b control in the graph? w stands for weight that the agent learned and b for beas