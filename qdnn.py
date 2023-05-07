
import numpy as np
from numpy import sum, subtract
from numpy.random import rand
from numpy.linalg import norm
from qiskit import Aer
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import HamiltonianGate, UnitaryGate

# Quantum Deep Neural Network Layer
class QDNNL():
    def __init__(self, num_qubits, D_epsilon, D_gamma, hamiltonians=None):
        if hamiltonians == None:
            hamiltonians = [np.eye(2 ** num_qubits)]
        self.num_qubits = num_qubits
        self.D_epsilon = D_epsilon
        self.D_gamma = D_gamma
        self.epsilon = 0.0
        self.learning_rate = 0.05
        self.hamiltonians = [UnitaryGate(h) for h in hamiltonians]
        # inputs from most recent forward propogation
        self.inputs = None
        # parameters to encoder, stored as one dimensional vector
        # size will be: num_qubits *(2 + self.D_epsilon * 3)
        self.encoder_parameters = list()
        # parameters of transformer circuit, stored as one dimensional vector
        self.transformer_parameters = [(rand() * 2 * np.pi) - np.pi for _ in range(self.num_qubits *(self.D_gamma * 3 + 2))]
        # self.transformer_parameters = np.zeros(self.num_qubits *(self.D_gamma * 3 + 2)) # TEMP
        # self.transformer_parameters[0] = np.pi
        # initialize quantum objects
        self.backend = Aer.get_backend('aer_simulator')
        
        # build encoder and transformer
        self.encoder = self._build_encoder()
        self.transformer = self._build_transformer()
        self._build_complete_circuits()
        
        self.current_results = list()
        
        
    def _build_complete_circuits(self):
        self.circuit : QuantumCircuit= QuantumCircuit(QuantumRegister(self.num_qubits), ClassicalRegister(self.num_qubits))
        self.circuit = self.circuit.compose(self.encoder,range(self.num_qubits))
        self.circuit = self.circuit.compose(self.transformer,range(self.num_qubits))
        self.complete_circuits = list()
        for hamiltonian in self.hamiltonians:
            self.complete_circuits.append(self.circuit.compose(self._build_measurement(hamiltonian), range(self.num_qubits)))
    
    def _build_ent(self):
        ent = QuantumCircuit(self.num_qubits)
        for qubit in range(1, self.num_qubits):
            ent.cx(qubit-1, qubit)
        ent.cx(self.num_qubits - 1, 0)
        return ent
        
        
        """
        create encoder circuit
        :param De: an integer that indicates how many times a part of this circuit is repeated
        :return: encoder quantum circuit
        :rtype: Instruction
        """
    def _build_encoder(self):
        encoder = QuantumCircuit(self.num_qubits)
        
        encoder.rx(np.pi, range(self.num_qubits))
        encoder.rz(np.pi, range(self.num_qubits))
        encoder.barrier(range(self.num_qubits))
        encoder.barrier(range(self.num_qubits))
        for _ in range(self.D_epsilon):
            encoder = encoder.compose(self._build_ent(), range(self.num_qubits))
            encoder.barrier(range(self.num_qubits))
            encoder.rz(np.pi, range(self.num_qubits))
            encoder.rx(np.pi, range(self.num_qubits))
            encoder.rz(np.pi, range(self.num_qubits))
            encoder.barrier(range(self.num_qubits))
            
        return encoder
    
    def _build_transformer(self):
        # function to return random parameter for initializing transformer or to assing params for after loss is calculated
        transformer = QuantumCircuit(self.num_qubits)
        
        
        # index of parameter in vector to refer to
        param_index = 0
        for _ in range(self.D_gamma):
            transformer = transformer.compose(self._build_ent(), range(self.num_qubits))
            transformer.barrier(range(self.num_qubits))
            for qb in range(self.num_qubits):
                transformer.rz(self.transformer_parameters[param_index], qb)
                param_index+=1
            for qb in range(self.num_qubits):
                transformer.rx(self.transformer_parameters[param_index], qb)
                param_index+=1
            for qb in range(self.num_qubits):
                transformer.rz(self.transformer_parameters[param_index], qb)
                param_index+=1
            transformer.barrier(range(self.num_qubits))
        for qb in range(self.num_qubits):
            transformer.rx(self.transformer_parameters[param_index], qb)
            param_index += 1
        for qb in range(self.num_qubits):
            transformer.rz(self.transformer_parameters[param_index], qb)
            param_index += 1
            
        return transformer
    
    def _build_measurement(self, hamiltonian):
        measurement = QuantumCircuit(self.num_qubits)
        measurement.append(hamiltonian, range(self.num_qubits))
        measurement.measure_all()
        return measurement
    
    def forward(self, input, epsilon_greedy = True):
        self.inputs = input
        if  (epsilon_greedy and rand() > self.epsilon) or not epsilon_greedy:
            # check that all input values are zeros or ones
            for i0, i1 in zip((input > 1), input < 0):
                assert i0 == False and i1 == False
            # initialize the state of the qubits in the encoder
            # execute job on simulator for each complete circuit
            results = list()
            for curr_complete_circuit_idx in range(len(self.complete_circuits)):
                
                curr_complete_circuit = QuantumCircuit(self.num_qubits)
                curr_complete_circuit.initialize(input)
                curr_complete_circuit = curr_complete_circuit.compose(self.complete_circuits[curr_complete_circuit_idx], range(self.num_qubits))
                curr_result_counts = self.backend.run(curr_complete_circuit).result().get_counts()
                results.extend(self._parse_result(curr_result_counts))
        else:
            return [rand() for _ in range(len(self.hamiltonians) * self.num_qubits)]
        return results
    
    def _parse_result(self, counts):
        """max_count = 0
        max_result = list(counts.keys())[0]
        for result,count in counts.items():
            if count > max_count:
                max_count = count
                max_result = result
        return [int(i) for i in max_result]
        """
        sum_counts = 0
        sum_results = np.zeros(len(list(counts.keys())[0]))
        for result,count in counts.items():
            sum_counts += count
            sum_results += [int(i) * count for i in result]
        average_results = sum_results / sum_counts
        average_norm = norm(average_results)
        if average_norm != 0:
            return average_results / average_norm
        else:
            return average_results
    def calculate_loss(self, output, target, type="MSE"):
        if type == "MSE":
            return np.sum(np.subtract(output, target) ** 2)/len(output)

    
    # this function only runs the transformer and is used for calculating loss for the transformer's parameters
    
    def _calculate_transformer_loss(self):
        
        dy_dw = list()
        # iterate through parameters in encoder
        for param_idx in range(len(self.transformer_parameters)):
            # calculate plus 
            self.transformer_parameters[param_idx] += np.pi / 2
            self.transformer = self._build_transformer()
            self._build_complete_circuits()
            h_plus = self.forward(self.inputs, epsilon_greedy=False)
            # calculate minus
            self.transformer_parameters[param_idx] -= np.pi
            self.transformer = self._build_transformer()
            self._build_complete_circuits()
            h_minus = self.forward(self.inputs, epsilon_greedy=False)
            # calculate differences between h_plus and h_minus values for each hamiltonian
            dy_dw_k = np.true_divide(subtract(h_plus,h_minus), 2)
            # restore parameter value and rebuild
            self.transformer_parameters[param_idx] += np.pi / 2
            self.transformer = self._build_transformer()
            self._build_complete_circuits()
            # add current param's lost to list of loss
            dy_dw.append(dy_dw_k)
        return dy_dw
    
    
    def backpropogate_error(self, loss):
        # calculate output loss w.r.t. input
        encoder_deltas = np.zeros(len(self.encoder_parameters))#  self._calculate_circuit_loss(self.encoder_parameters)
        # iterate thorugh inputs, assign gradient for each
        transformer_gradients = self._calculate_transformer_loss()
        # loss with respect to inputs
        loss_inputs = loss * encoder_deltas
        # loss with respect to parameters
        loss_parameters = [loss * grad for grad in transformer_gradients]
        return loss_inputs, loss_parameters
    
    
    def update_weights(self, transformer_gradients):
        for grad_idx in range(len(transformer_gradients)):
            self.transformer_parameters[grad_idx] -= self.learning_rate * self.calculate_loss(transformer_gradients[grad_idx], self.current_results)
        self._build_transformer()
        self._build_complete_circuits()
        
        
class InputLayer(QDNNL):
    def __init__(self, num_qubits, D_epsilon, D_gamma):
        super().__init__(num_qubits, D_epsilon, D_gamma)
        # prepare quantum circuit
        self.input_layer = QuantumCircuit(QuantumRegister(num_qubits), ClassicalRegister(num_qubits))
        # add encoder circuit to layer
        self.input_layer = self.input_layer.compose(self._build_encoder(),range(self.num_qubits))
        self.input_layer = self.input_layer.compose(self._build_transformer(),range(self.num_qubits))
        


class QDNN():
     
    def __init__(self) -> None:
        # fixed theta used in encoding circuits
        self.theta = np.pi / 4
        self.input_layer = QDNNL()
