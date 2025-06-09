import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('Optical_Flow_tinycmax/data/flow_maps_for_trajectory.npy', 'rb') as f:
    flow_sequence = pickle.load(f)

print(len(flow_sequence))

def mean_flow(flow):
    flow_x, flow_y = flow[0], flow[1]
    mean_x, mean_y = flow_x.mean(), flow_y.mean()

    return np.array([mean_x, mean_y])

flow_seq = []
for flow in flow_sequence:
    avg_vec = mean_flow(flow)
    flow_seq.append(avg_vec)

flow_array = np.array(flow_seq)  # shape: (512, 2)
u_measured = flow_array[:, 0]
v_measured = flow_array[:, 1]
time = np.arange(len(flow_seq)) 

class PIController:
    def __init__(self, Kp, Ki, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral = 0.0

    def update(self, error):
        self.integral += error * self.dt
        return self.Kp * error + self.Ki * self.integral
    

# Desired optical flow velocities
u_desired = 1.0  # pixels/sec
v_desired = 1.0

# Tuning parameters 
Kp = 1.1
Ki = 0.1
dt = 0.01  

# Create controllers
roll_controller = PIController(Kp, Ki, dt)  # u
pitch_controller = PIController(Kp, Ki, dt)  # v

# Output storage
roll_cmds = []
pitch_cmds = []
u_corrected = []
v_corrected = []

for i in range(len(flow_seq)):
    u_error = u_desired - u_measured[i]
    v_error = v_desired - v_measured[i]

    roll_cmd = roll_controller.update(u_error)
    pitch_cmd = pitch_controller.update(v_error)

    roll_cmds.append(roll_cmd)
    pitch_cmds.append(pitch_cmd)

    # Simulate corrected u/v (measured + control)
    u_controlled = u_measured[i] + roll_cmd
    v_controlled = v_measured[i] + pitch_cmd

    u_corrected.append(u_controlled)
    v_corrected.append(v_controlled)

plt.figure(figsize=(12, 6))

# Plot u flow
plt.subplot(2, 1, 1)
plt.plot(time, u_measured, label="Measured u", linestyle='--', color='red')
plt.plot(time, u_corrected, label="Controlled u", color='blue')
plt.axhline(u_desired, color='black', linestyle=':', label="Target u")
plt.title("Roll Control via PI Controller (u)")
plt.ylabel("u (pixels/unit time)")
plt.grid(True)
plt.legend()

# Plot v flow
plt.subplot(2, 1, 2)
plt.plot(time, v_measured, label="Measured v", linestyle='--', color='red')
plt.plot(time, v_corrected, label="Controlled v", color='green')
plt.axhline(v_desired, color='black', linestyle=':', label="Target v")
plt.title("Pitch Control via PI Controller (v)")
plt.ylabel("v (pixels/unit time)")
plt.xlabel("Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('pi_controller.png', dpi=300, bbox_inches='tight')