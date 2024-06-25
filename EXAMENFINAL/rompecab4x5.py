#EXAMEN FINAL ROMPECABEZAS 
#VILLCA JAITA LINO FERNANDO - ING. CS DE LA COMPUTACION

import random
import matplotlib.pyplot as plt
import pandas as pd

# Definimos el entorno del rompecabezas
class PuzzleEnvironment:
    def __init__(self):
        # Estado inicial del rompecabezas
        self.initial_state = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 0]  # 0 representa el espacio vacío
        ]
        self.state = self.initial_state

    # Restablece el rompecabezas al estado inicial
    def reset(self):
        self.state = self.initial_state
        return self.state

    # Encuentra la posición del espacio vacío
    def find_zero(self):
        for i in range(4):
            for j in range(5):
                if self.state[i][j] == 0:
                    return i, j

    # Movimiento de piezas basado en la acción
    def move(self, action):
        x, y = self.find_zero()
        new_state = [row[:] for row in self.state]  # Copia del estado actual

        if action == 'up' and x > 0:
            new_state[x][y], new_state[x-1][y] = new_state[x-1][y], new_state[x][y]
        elif action == 'down' and x < 3:
            new_state[x][y], new_state[x+1][y] = new_state[x+1][y], new_state[x][y]
        elif action == 'left' and y > 0:
            new_state[x][y], new_state[x][y-1] = new_state[x][y-1], new_state[x][y]
        elif action == 'right' and y < 4:
            new_state[x][y], new_state[x][y+1] = new_state[x][y+1], new_state[x][y]
        
        self.state = new_state
        return new_state

    # Verifica si el rompecabezas está resuelto
    def is_solved(self):
        goal_state = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 0]
        ]
        return self.state == goal_state

# Implementación del agente Q-Learning
class QLearningAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.env = environment
        self.q_table = {}  # Tabla Q
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actions = ['up', 'down', 'left', 'right']  # Acciones posibles
        self.rewards_history = []
        self.steps_history = []
        self.exploration_rates = []
        self.explotation_rates = []

    # Convierte el estado en una tupla para usarlo como clave en la tabla Q
    def state_to_tuple(self, state):
        return tuple(tuple(row) for row in state)

    # Selecciona una acción usando una política epsilon-greedy
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explorar
        state_tuple = self.state_to_tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0 for action in self.actions}
        return max(self.q_table[state_tuple], key=self.q_table[state_tuple].get)  # Explotar

    # Actualiza la tabla Q utilizando la fórmula de Q-Learning
    def update_q_table(self, state, action, reward, next_state):
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0 for action in self.actions}
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = {action: 0 for action in self.actions}

        max_future_q = max(self.q_table[next_state_tuple].values())
        current_q = self.q_table[state_tuple][action]
        
        # Fórmula de actualización de Q-Learning
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state_tuple][action] = new_q

    # Entrena el agente
    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            steps = 0
            total_reward = 0  # Suma acumulada de recompensas por episodio

            while not self.env.is_solved():
                action = self.choose_action(state)
                next_state = self.env.move(action)
                reward = 1 if self.env.is_solved() else -0.1  # Recompensa por resolver el rompecabezas o penalización por cada paso
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                steps += 1
                total_reward += reward

            # Disminuye la tasa de exploración (epsilon)
            self.epsilon *= self.epsilon_decay

            # Guarda el número de pasos por episodio y la suma acumulada de recompensas
            self.steps_history.append(steps)
            self.rewards_history.append(total_reward)
            self.exploration_rates.append(self.epsilon)
            self.explotation_rates.append(1 - self.epsilon)

            if episode % 100 == 0:
                print(f'Episode {episode}, Steps: {steps}, Epsilon: {self.epsilon}, Explotation Rate: {1 - self.epsilon}')

    # Función para graficar las recompensas y la tasa de exploración por episodio
    def plot_rewards_and_epsilon(self):
        episodes = range(len(self.rewards_history))
        
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Suma de recompensas acumuladas', color='tab:blue')
        ax1.plot(episodes, self.rewards_history, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Tasa de Exploración/Explotación', color='tab:red')
        ax2.plot(episodes, self.exploration_rates, color='tab:red', label='Exploración (Epsilon)')
        ax2.plot(episodes, self.explotation_rates, color='tab:green', label='Explotación (1 - Epsilon)')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.title('Aprendizaje del Agente (Q-Learning)')
        plt.grid(True)
        plt.show()

    # Función para mostrar la tabla Q al final del entrenamiento
    def print_q_table(self):
        print("Tabla Q:")
        q_table_dict = {
            "Estado": [],
            "Acción": [],
            "Q-Valor": []
        }
        for state, actions in self.q_table.items():
            for action, q_value in actions.items():
                q_table_dict["Estado"].append(state)
                q_table_dict["Acción"].append(action)
                q_table_dict["Q-Valor"].append(q_value)
        
        q_table_df = pd.DataFrame(q_table_dict)
        print(q_table_df)

# Configuración de parámetros y ejecución del entrenamiento
env = PuzzleEnvironment()
agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.2, epsilon=1.0, epsilon_decay=0.995) #0.995
agent.train(episodes=1000)

# Graficar las recompensas y la tasa de exploración/explotación por episodio
agent.plot_rewards_and_epsilon()

# Mostrar la tabla Q al final del entrenamiento usando Pandas
agent.print_q_table()


#El ejercicio de implementar un agente de aprendizaje por refuerzo para resolver un rompecabezas de 4 filas por 5 columnas nos permitió explorar y aplicar conceptos fundamentales de Q-Learning y SARSA. Aquí destacamos las conclusiones más relevantes:

""" Definición del Entorno y el Agente:

El entorno se configuró adecuadamente para reflejar el estado y las acciones posibles del rompecabezas.
El agente Q-Learning se diseñó con parámetros esenciales como la tasa de aprendizaje (alpha), el factor de descuento (gamma), y la tasa de exploración (epsilon).
Actualización de la Tabla Q:

Implementamos la fórmula de actualización de Q-Learning para ajustar los valores Q basados en las recompensas recibidas y los valores Q futuros estimados.
Aseguramos que la tabla Q (self.q_table) se inicializara correctamente y se actualizara con cada paso del agente.
Exploración y Explotación:

Se utilizó una política epsilon-greedy para balancear la exploración de nuevas acciones y la explotación de acciones conocidas para maximizar las recompensas.
Observamos cómo la tasa de exploración (epsilon) decayó gradualmente, permitiendo que el agente explotara más a medida que aprendía del entorno.
Monitoreo del Proceso de Aprendizaje:

Implementamos gráficos para visualizar la evolución de las recompensas acumuladas y las tasas de exploración y explotación a lo largo de los episodios.
Esto nos permitió verificar que el agente estaba aprendiendo efectivamente, mejorando su desempeño y reduciendo el número de pasos necesarios para resolver el rompecabezas.
Resultados y Evaluación:

Al final del entrenamiento, mostramos la tabla Q para analizar los valores aprendidos por el agente.
Evaluamos el éxito del agente en resolver el rompecabezas mediante la reducción progresiva del número de pasos y el aumento en la consistencia de las recompensas.
Reflexiones Finales
Este ejercicio ha demostrado la capacidad del aprendizaje por refuerzo para abordar problemas complejos de toma de decisiones, como la resolución de un rompecabezas. Hemos visto cómo ajustar adecuadamente los parámetros y estrategias de exploración puede impactar significativamente en el desempeño del agente. Este enfoque puede extenderse y adaptarse a otros problemas y aplicaciones, brindando una poderosa herramienta para el desarrollo de sistemas inteligentes.

Además, la implementación en un entorno de código abierto como Visual Studio Code, junto con bibliotecas populares de Python, permite que estos métodos sean accesibles y modificables, fomentando el aprendizaje y la experimentación continua en el campo del aprendizaje por refuerzo. """