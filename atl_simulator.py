import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def entropy(p):
    return -np.sum(p * np.log(p + 1e-9))

def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-9) / (q + 1e-9)))

# Inicialización según el COM
np.random.seed(42)
vocab_size, timesteps = 20, 100
p_logits = np.random.randn(vocab_size)
q_logits = np.random.randn(vocab_size) * 0.5
T_values, E_values = [], []
lambda_reg, E_target = 0.1, 0.5

for t in range(timesteps):
    if t % 20 == 0 and t > 0:
        p_logits += np.random.normal(0, 2.0, vocab_size) # Perturbación OOD
    
    p, q = softmax(p_logits), softmax(q_logits)
    E_t = kl_divergence(p, q)
    T_t = E_t / (entropy(p) + 1e-9)
    
    E_values.append(E_t)
    T_values.append(T_t)
    
    # Mecanismo de regulación (Pólux)
    q_logits += lambda_reg * (E_t - E_target) * (p - q)
    p_logits += np.random.normal(0, 0.1, vocab_size)

print("Simulación ATL/COM completada con éxito.")
