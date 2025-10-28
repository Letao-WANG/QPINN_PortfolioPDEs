import pennylane as qml

dev = qml.device("lightning.gpu", wires=10)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], 0)
    qml.RY(params[1], 1)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

print(circuit([0.2, 0.1]))
