import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt

X = []
y = []

def get_data_gui(veri):

    satir = veri.split("\n")
    satir = satir[1:-1] #başlık silmek için kodumuz

    for veri in satir:

        satir_ayrilmis = veri.strip().split(",")
        X.append(satir_ayrilmis[0].split())
        y.append(satir_ayrilmis[1].split())

    X_np = np.array(X)
    y_np = np.array(y)

    return X_np.astype(float), y_np.astype(float)

def train_test_ayrimi(oran = 0.1, X=None, y=None):

    test_seti_adet = int(len(X) * oran)
    test_indexes = random.sample(range(0, len(X)), test_seti_adet)
    train_indexes = [x for x in range(0, len(X)) if x not in test_indexes]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]

def normalizasyon_islemi(X, y):

    min_deger = np.min(X)
    max_deger = np.max(X)
    normalize_edilmiş_veri_X = (X - min_deger) / (max_deger - min_deger)

    min_deger = np.min(y)
    max_deger = np.max(y)
    normalize_edilmiş_veri_y = (y - min_deger) / (max_deger - min_deger)

    return normalize_edilmiş_veri_X, normalize_edilmiş_veri_y

class ArtificialNeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, activation_function):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.biases_hidden = np.random.rand(self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.biases_output = np.random.rand(self.output_size)

    def tanH(self, data):
        return np.tanh(data)
    def ReLU(self, data):
        return np.maximum(0,data)
    def sigmoid(self, data):
        return 1 / (1+np.exp(-data))

    def activation_func(self, data):

        if self.activation_function == "tanh":
            return self.tanH(data)
        if self.activation_function == "relu":
            return self.ReLU(data)
        if self.activation_function == "sigmoid":
            return self.sigmoid(data)
        else:
            print("False activation function")


    def forward(self, data):

        hidden_input = np.dot(data, self.weights_input_hidden) + self.biases_hidden
        hidden_output = self.activation_func(hidden_input)

        final_inputs = np.dot(hidden_output, self.weights_hidden_output) + self.biases_output
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

class ObjectiveFunction:

    def __init__(self, ArtificialNeuralNetwork, datasets):
        self.artificialNeuralNetwork = ArtificialNeuralNetwork
        self.datasets = datasets

    def __len__(self):
        # Return the number of parameters in the neural network
        return len(self.artificialNeuralNetwork.weights_input_hidden.flatten()) + \
            len(self.artificialNeuralNetwork.biases_hidden) + \
            len(self.artificialNeuralNetwork.weights_hidden_output.flatten()) + \
            len(self.artificialNeuralNetwork.biases_output)

    def __call__(self, params):
        self.set_params(params)
        total_error = 0

        for i in range(len(self.datasets)):
            predictions = self.artificialNeuralNetwork.forward(self.datasets[0][i])
            error = np.mean((predictions - self.datasets[1][i]) ** 2)
            print(error)
            total_error += error

        return total_error


    def set_params(self, params):
        start = 0
        end = self.artificialNeuralNetwork.input_size * self.artificialNeuralNetwork.hidden_size
        self.artificialNeuralNetwork.weights_input_hidden = np.reshape(params[start:end],(self.artificialNeuralNetwork.input_size, self.artificialNeuralNetwork.hidden_size))
        start = end
        end = start + self.artificialNeuralNetwork.hidden_size
        self.artificialNeuralNetwork.biases_hidden = params[start:end]
        start = end
        end = start + self.artificialNeuralNetwork.hidden_size * self.artificialNeuralNetwork.output_size
        self.artificialNeuralNetwork.weights_hidden_output = np.reshape(params[start:end],(self.artificialNeuralNetwork.hidden_size, self.artificialNeuralNetwork.output_size))
        start = end
        end = start + self.artificialNeuralNetwork.output_size
        self.artificialNeuralNetwork.biases_output = params[start:end]

def abc_algorithm(objective_function, n_iterations=100, n_employed_bees=30, limit=float('inf')):
    best_solution = None
    best_fitness = float('inf')
    evaluations = 0

    for iteration in range(n_iterations):
        employed_bees = 2 * np.random.rand(n_employed_bees, len(objective_function)) - 1

        # Evaluate fitness for employed bees
        for i in range(n_employed_bees):
            if evaluations >= limit:
                return best_solution, best_fitness

            params = employed_bees[i]
            fitness = objective_function(params)
            evaluations += 1

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = params.copy()

    return best_solution, best_fitness


def pso_algorithm(objective_function, n_iterations=100, n_particles=30, inertia=0.5,
                  cognitive_constant=1.5, social_constant=1.5, limit=float('inf')):

    particles = 2 * np.random.rand(int(n_particles), len(objective_function)) - 1
    velocities = np.random.rand(int(n_particles), len(objective_function)) - 0.5

    personal_best_positions = particles.copy()
    personal_best_fitness = np.array([objective_function(p) for p in particles])
    global_best_position = particles[np.argmin(personal_best_fitness)]
    global_best_fitness = np.min(personal_best_fitness)

    evaluations = len(particles)  # Initial evaluations

    for iteration in range(n_iterations):
        for i in range(n_particles):
            if evaluations >= limit:
                return global_best_position, global_best_fitness


            r1, r2 = np.random.rand(2)
            velocities[i] = (inertia * velocities[i] +
                             cognitive_constant * r1 *
                             (personal_best_positions[i] - particles[i]) +
                             social_constant * r2 * (global_best_position - particles[i]))

            particles[i] = particles[i] + velocities[i]
            fitness = objective_function(particles[i])
            evaluations += 1

            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i].copy()

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles[i].copy()

    return global_best_position, global_best_fitness

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

def forward_pass(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2
def backward_pass(X, y, z1, a1, z2, a2, w1, b1, w2, b2, learning_rate):
    output_error = a2 - y
    output_delta = output_error * sigmoid_derivative(a2)

    hidden_error = np.dot(output_delta, w2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)

    w2 -= np.dot(a1.T, output_delta) * learning_rate
    b2 -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    w1 -= np.dot(X.T, hidden_delta) * learning_rate
    b1 -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    return w1, b1, w2, b2


def back_propagation(X, y, hidden_size, n_iterations=10000, learning_rate=0.1):
    input_size = X.shape[1]
    output_size = y.shape[1]
    w1, b1, w2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for iteration in range(n_iterations):
        z1, a1, z2, a2 = forward_pass(X, w1, b1, w2, b2)
        w1, b1, w2, b2 = backward_pass(X, y, z1, a1, z2, a2, w1, b1, w2, b2, learning_rate)

        if iteration % 1000 == 0:
            loss = np.mean(np.square(y - a2))

        print(loss)

    bp_best_fitness = loss
    bp_best_solution = np.concatenate([w1.flatten(), b1.flatten(), w2.flatten(), b2.flatten()])
    return bp_best_fitness, bp_best_solution




file_content = ""

def select_file(entry):

    global file_content
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
            entry.delete(0, tk.END)
            entry.insert(0, file_content)


def store_ann_inputs():
    global ann_input_size, ann_hidden_size, ann_output_size, ann_data, ann_test_size
    ann_input_size = float(ann_entries[0].get())
    ann_hidden_size = float(ann_entries[1].get())
    ann_output_size = float(ann_entries[2].get())
    ann_data = ann_entries[3].get()
    ann_test_size = float(ann_entries[4].get())

def store_abc_inputs():

    results_matrix = []

    ann = ArtificialNeuralNetwork(input_size=int(ann_entries[0].get()),hidden_size=int(ann_entries[1].get()),
                        output_size= int(ann_entries[2].get()),activation_function=str(ann_entries[5].get()))

    X, y = get_data_gui(ann_entries[3].get())
    X_norm, y_norm = normalizasyon_islemi(X, y)
    X_train, X_test, y_train, y_test = train_test_ayrimi(oran=float(ann_entries[4].get()),
                                                         X=X_norm, y = y_norm)
    print("X_train for ABC : ", X_train)
    obj = ObjectiveFunction(ArtificialNeuralNetwork=ann, datasets=(X_train, y_train))

    for i in range(int(abc_entries[3].get())):

        abc_best_solution, abc_best_fitness = abc_algorithm(objective_function=obj,
                                                        n_iterations=int(abc_entries[1].get()),
                                                        n_employed_bees=int(abc_entries[0].get()),
                                                        limit=int(abc_entries[2].get()))

        results_matrix.append(abc_best_fitness)

    print("----------------ABC RESULTS---------------- \nBest Fitness :", abc_best_fitness,
          "\nBest weights : ", abc_best_solution,
          "\nAverage : ", np.mean(np.array(results_matrix)))

    result_msg = f"Best Fitness : {abc_best_fitness} \n Best Weights {abc_best_solution} \n Average : {np.mean(np.array(results_matrix))}"
    messagebox.showinfo("ABC Results", message=result_msg)

def store_pso_inputs():

    results_matrix = []

    ann = ArtificialNeuralNetwork(input_size=int(ann_entries[0].get()),
                                  hidden_size=int(ann_entries[1].get()),
                                  output_size=int(ann_entries[2].get()),
                                  activation_function=str(ann_entries[5].get()))

    X, y = get_data_gui(ann_entries[3].get())
    X_norm, y_norm = normalizasyon_islemi(X, y)
    X_train, X_test, y_train, y_test = train_test_ayrimi(oran=float(ann_entries[4].get()),
                                                         X=X_norm, y=y_norm)
    obj = ObjectiveFunction(ArtificialNeuralNetwork=ann, datasets=(X_train, y_train))

    for i in range(int(pso_entries[6].get())):

        pso_best_solution, pso_best_fitness = pso_algorithm(objective_function=obj,
                    n_iterations=int(pso_entries[0].get()), n_particles=int(pso_entries[1].get()),
                    inertia=float(pso_entries[2].get()), cognitive_constant=float(pso_entries[3].get()),
                    social_constant=float(pso_entries[5].get()), limit=int(pso_entries[4].get())                                                        )

        results_matrix.append(pso_best_fitness)

    print("----------------PSO RESULTS---------------- \nBest Fitness :", pso_best_fitness,
          "\nBest weights : ", pso_best_solution,
          "\nAverage : ", np.mean(np.array(results_matrix)))

    result_msg = f"Best Fitness : {pso_best_fitness} \nBest Weights {pso_best_solution} \n Average : {np.mean(np.array(results_matrix))}"
    messagebox.showinfo("PSO Results", result_msg)


def store_bp_inputs():

    ann = ArtificialNeuralNetwork(input_size=int(ann_entries[0].get()),hidden_size=int(ann_entries[1].get()),
                    output_size=int(ann_entries[2].get()),activation_function=str(ann_entries[5].get()))

    X, y = get_data_gui(ann_entries[3].get())
    X_norm, y_norm = normalizasyon_islemi(X, y)
    X_train, X_test, y_train, y_test = train_test_ayrimi(oran=float(ann_entries[4].get()),
                                                         X=X_norm, y=y_norm)
    obj = ObjectiveFunction(ArtificialNeuralNetwork=ann, datasets=(X_train, y_train))
    bp_best_fitness, bp_best_solution = back_propagation(X_train,
                               y_train,
                               int(ann_entries[1].get()),
                               int(bp_entries[1].get()),
                               float(bp_entries[0].get()))

    result_msg = f"Best Fitness : {bp_best_fitness} \nBest Weights {bp_best_solution}"
    messagebox.showinfo("Back Prograpation Results", result_msg)

    print("BP RESULTS : \n", "Best Fitness : " ,bp_best_fitness, "\nBest Solution :", bp_best_solution)

# Create the main window
root = tk.Tk()
root.title("Artificial Neural Network Tool")

# Create a frame for each section
frame1 = ttk.Frame(root, padding="10")
frame1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

frame2 = ttk.Frame(root, padding="10")
frame2.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

frame3 = ttk.Frame(root, padding="10")
frame3.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

frame4 = ttk.Frame(root, padding="10")
frame4.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

# Add a header for each section
ttk.Label(frame1, text="Artificial Neural Network", font=("Helvetica", 16)).grid(row=0, columnspan=3, pady=10)
ttk.Label(frame2, text="Artificial Bee Colony", font=("Helvetica", 16)).grid(row=0, columnspan=2, pady=10)
ttk.Label(frame3, text="Particle Swarm", font=("Helvetica", 16)).grid(row=0, columnspan=2, pady=10)
ttk.Label(frame4, text="Back Propagation", font=("Helvetica", 16)).grid(row=0, columnspan=2, pady=10)

# Helper function to create labels and entries
def create_label_entry(frame, text, row, colspan=1):
    ttk.Label(frame, text=text).grid(row=row, column=0, padx=5, pady=5, sticky="e")
    entry = ttk.Entry(frame)
    entry.grid(row=row, column=1, padx=5, pady=5, columnspan=colspan, sticky="w")
    return entry

# Artificial Neural Network inputs
ann_inputs = ["Input Size", "Hidden Size", "Output Size", "Data", "Test Size", "Activation Function"]

ann_entries = []
for i, label in enumerate(ann_inputs, 1):
    if label == "Data":
        entry = create_label_entry(frame1, label, i, 1)
        button = ttk.Button(frame1, text="Browse", command=lambda e=entry: select_file(e))
        button.grid(row=i, column=2, padx=5, pady=5, sticky="w")
        ann_entries.append(entry)
    else:
        ann_entries.append(create_label_entry(frame1, label, i))

# Artificial Bee Colony inputs
abc_inputs = ["Employed Bees", "Iteration", "Limit", "Training Count"]
abc_entries = []
for i, label in enumerate(abc_inputs, 1):
    abc_entries.append(create_label_entry(frame2, label, i))

# Run ABC button
ttk.Button(frame2, text="Run ABC", command=store_abc_inputs).grid(row=len(abc_inputs) + 1, columnspan=2, pady=10)

# Particle Swarm Optimization inputs
pso_inputs = ["Iteration", "Particles", "Inertia", "Cog", "Limit", "Soc", "Training Count"]
pso_entries = []
for i, label in enumerate(pso_inputs, 1):
    pso_entries.append(create_label_entry(frame3, label, i))

# Run PSO button
ttk.Button(frame3, text="Run PSO", command=store_pso_inputs).grid(row=len(pso_inputs) + 1, columnspan=2, pady=10)

# Back Propagation inputs
bp_inputs = ["Learning Rate", "Iteration"]
bp_entries = []
for i, label in enumerate(bp_inputs, 1):
    bp_entries.append(create_label_entry(frame4, label, i))

# Run Back Propagation button
ttk.Button(frame4, text="Run Back Propagation", command=store_bp_inputs).grid(row=len(bp_inputs) + 1, columnspan=2, pady=10)

# Run the application
root.mainloop()

