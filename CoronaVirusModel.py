import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
## SEIR model of population disease spread:
##          Coronavirus in Wuhan case study
##            Written by Jonathan Klaric

## References:
## SIR/SEIR mathematical epidemiology models:
## https://en.wikipedia.org/.../Compartmental_models_in...
## SIR in python example:
## https://scipython.com/.../additio.../the-sir-epidemic-model/


## ASSUMPTIONS:
# The population is fully 'mixed' (anyone can meet, interact with anyone)
# Every person is equally likely to get the disease
# Every person is equally likely to transmit the disease


## state variables
# N = Total population
# S = Susceptible
# E = Exposed
# I = infected
# R = Recovered/removed. Either dead or no longer contagious and now immune.


## Parameters
# mu[0] = natural birth rate
# mu[1] = natural death rate
# beta = infectious rate of the disease
# = average number of contacts per person * probability of transmission
# gamma = recovery or removal rate. If D is duration once infection presents, gamma = 1/D
# alpha = incubation period. If incubation is duration D, then alpha = 1/D
# kappa = infectious rate of the disease during incubation period
# = average number of contacts per person (in incubation period) * probability of transmission


## Dynamical equations:
# dS/dt = mu[0]*N - mu[1]*S - beta*I*S/N - kappa*E*S/N
# dE/dt = beta*I*S/N + kappa*E*S/N - (mu[1]+alpha)*E
# dI/dt = alpha*E - (mu[1]+gamma)*I
# dR/dt = gamam*I - mu[1]*R


def deriv(state, t, N, mu, alpha, beta, gamma, kappa):
    S, E, I, R = state
    dSdt = mu[0]*N - mu[1]*S - beta*I*S/N - kappa*E*S/N
    dEdt = beta*I*S/N + kappa*E*S/N - (mu[1]+alpha)*E
    dIdt = alpha*E - (mu[1]+gamma)*I
    dRdt = gamma*I - mu[1]*R
    return dSdt, dEdt, dIdt, dRdt


# initialize:
N = 53000000 # population of Wuhan
E0 = 1 # number of initially exposed people
S, E, I, R = N-E0, E0, 0, 0
t = np.arange(0,190,0.1)
intial_state = S, E, I, R
## transition rates between stages of the disease
alpha = 1/14.0 # 14 day incubation period
gamma = 1/7.0 # 7 day infectious period before death/recovery
## transmission rates = average number of contacts per person * probability of transmission

# incubation period
av_contacts = 3
prob_of_trans = 0.1
kappa = av_contacts*prob_of_trans
# infectious period
av_contacts = 1
prob_of_trans = 0.0005
beta = av_contacts*prob_of_trans
# birth and death rates
mu = [0.0,0.0] # assume pop. constant in this scenario

# solve the ODE system
ret = odeint(deriv, intial_state, t, args=(N, mu, alpha, beta, gamma, kappa))
S, E, I, R = ret.T
total_exposed = E+I

# print some summary information
print("Total who got the virus: {:,.0f}".format(R[-1]))
print("Total who never got the virus: {:,.0f}".format(S[-1]))
print("Peak number of simultaneously contageous people: {:,.0f}".format(np.max(total_exposed)))
print("Peak number of simultaneously symptomatic people (i.e. peak number requiring hospitalization): {:,.0f}".format(np.max(I)))

# plotting functions
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/1000000, 'g', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, total_exposed/1000000, 'y', alpha=0.5, lw=2, label='Exposed+Infected')
ax.plot(t, I/1000000, 'r', alpha=0.5, lw=2, label='Infected & symptomatic')
ax.plot(t, R/1000000, 'b', alpha=0.5, lw=2, label='Recovered')
plt.title("Spread of coronavirus in Wuhan")
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number (millions)')
#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
