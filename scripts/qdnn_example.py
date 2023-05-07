import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import Aer, transpile
from qiskit.extensions import HamiltonianGate
import numpy as np
from qdnn.qdnn import QDNNL
from matplotlib import pyplot as plt
# build hamiltonians
from tqdm import tqdm
# construct layer
input_layer = QDNNL(4, 1, 1)
input = np.zeros(16)
input[1] = 1

losses = list()
target = np.array([0,1,0,0])
for i in tqdm(range(200)):
    input_layer.epsilon *= 0.95
    results = input_layer.forward(input)
    input_layer.current_results = results
    loss = input_layer.calculate_loss(results, target)
    loss_inputs, loss_parameters = input_layer.backpropogate_error(loss)
    input_layer.update_weights(loss_parameters)
    losses.append(loss)
plt.xlabel("timestep")
plt.ylabel("loss")
plt.plot(range(len(losses)), losses)
plt.show()
print(input_layer.complete_circuits[0].draw())
# for result in results:
#     print(result)
