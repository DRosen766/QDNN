import qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import Aer, transpile
from qiskit.extensions import HamiltonianGate
from rfrl_gym.agents.qdnn import *



# build hamiltonians

# construct layer
input_layer = QDNNL(4, 0, 0)
input = np.zeros(16)
input[0] = 1

target = np.array([0,1,0,0])
for i in range(500):
    input_layer.epsilon *= 0.95
    results = input_layer.forward(input)
    input_layer.current_results = results
    loss = input_layer.calculate_loss(results, target)
    loss_inputs, loss_parameters = input_layer.backpropogate_error(loss)
    input_layer.update_weights(loss_parameters)
    print(loss, results)
    # print(input_layer.transformer_parameters, "\n")
print(input_layer.complete_circuits[0].draw())
# for result in results:
#     print(result)
