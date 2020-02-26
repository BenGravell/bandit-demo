import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

# Random seed for repeatability
seed = 1
npr.seed(seed)

# Number of arms of the bandit
k = 10

# Distribution limits of the rewards
s1 = npr.rand(k)/2
s2 = (1+npr.rand(k))/2

# Reward
def reward(a):
    r = s1[a] + (s2[a]-s1[a])*npr.rand()
    return r

# True means
q_opt = (s1+s2)/2

# Optimal action
a_opt = np.argmax(q_opt)

# Epsilon-greedy policy
def epsilon_greedy(q, t, n, epsilon=0.1):
    a = np.argmax(q)
    if npr.rand() < epsilon:
        i = npr.randint(k-1)
        a = np.arange(k)[i if i < a else i+1]
    return a

# Upper confidence bound policy
def upper_confidence_bound(q, t, n, c=1.0):
    a = np.argmax(q + c*(2*np.log(t)/(n+1))**0.5)
    return a

# Number of trials
nt = 100000

def experiment(policy, parameter):
    # Initialize
    q = np.zeros(k)
    n = np.zeros(k)

    # History
    a_hist = np.zeros(nt)
    r_hist = np.zeros(nt)
    q_hist = np.zeros([nt, k])
    n_hist = np.zeros([nt, k])
    t_hist = np.arange(nt)

    # Iterate
    for t in t_hist:
        # Generate action
        a = policy(q, t+1, n, parameter)

        # Generate reward
        r = reward(a)

        # Accumulate number of plays of chosen action
        n[a] += 1

        # Record history
        a_hist[t] = a
        r_hist[t] = r
        q_hist[t] = q
        n_hist[t] = n

        # Update action-value function using incremental sample-average rule
        q[a] += (r - q[a])/n[a]

    return q, a_hist, r_hist, q_hist, n_hist, t_hist

experiment_data_list = []
policy_list = [epsilon_greedy, epsilon_greedy, epsilon_greedy, upper_confidence_bound]
parameter_list = [0.002, 0.02, 0.2, 1.00]
policy_strings = ['$\epsilon$-greedy, $\epsilon=0.002$',
                  '$\epsilon$-greedy, $\epsilon=0.020$',
                  '$\epsilon$-greedy, $\epsilon=0.200$',
                  'upper confidence bound']

for policy, parameter in zip(policy_list, parameter_list):
    experiment_data_list.append(experiment(policy, parameter))

# Plotting
plt.close('all')

x = s1+(s2-s1)*np.linspace(0, 1)[:, None]
bp = plt.boxplot(x, whis='range')
plt.xlabel('Arm')
plt.ylabel('Reward')
plt.title('Bandit rewards')
plt.show()

num_plots = 3
fig_list, ax_list = [], []
for i in range(num_plots):
    fig, ax = plt.subplots()
    fig_list.append(fig)
    ax_list.append(ax)

for i, policy_string in enumerate(policy_strings):
    q, a_hist, r_hist, q_hist, n_hist, t_hist = experiment_data_list[i]

    # Reward history
    ax_list[0].plot(t_hist, np.cumsum(r_hist)/(t_hist+1))

    # Expected regret history
    r_hist_expected = np.array([q_opt[a] for a in a_hist.astype(int)])
    regret = q_opt.max() - r_hist_expected
    ax_list[1].plot(t_hist, np.cumsum(regret))

    # Action-value function estimation error
    ax_list[2].semilogy(t_hist, la.norm(q_hist-q_opt, axis=1))

ax_list[0].plot(t_hist, np.max(q_opt)*np.ones(nt), 'k--')

title_str_list = ['Time-averaged total reward vs time',
                  'Total expected regret vs time',
                  'Action-value estimation error vs time']
for ax, title_str in zip(ax_list, title_str_list):
    ax.set_title(title_str)
    ax.legend(policy_strings)
plt.show()