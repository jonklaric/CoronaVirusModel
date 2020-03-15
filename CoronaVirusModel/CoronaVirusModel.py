import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## SEIR model of population disease spread:
##      Coronavirus in Wuhan case study
##      Written by Jonathan Klaric
## References:
##      SIR/SEIR mathematical epidemiology models: 
##          https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model
##      SIR in python example:
##          https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

## ASSUMPTIONS:
#   The population is fully 'mixed' (anyone can meet, interact with anyone)
#   Every person is equally likely to get the disease
#   Every person is equally likely to transmit the disease


## state variables
# N = Total population
# S = Susceptible
# E = Exposed
# I = infected
# R = Recovered/removed. Either dead or no longer contagious and now immune.

## parameters
# mu[0]     = natural birth rate
# mu[1]     = natural death rate
# beta      = infectious rate of the disease 
#           = average number of contacts per person * probability of transmission
# gamma     = recovery or removal rate. If D is duration once infection presents, gamma = 1/D
# alpha     = incubation period. If incubation is duration D, then alpha = 1/D
# kappa     = infectious rate of the disease during incubation period
#           = average number of contacts per person (in incubation period) * probability of transmission



## Dynamical equations (SIR model):

# dS/dt = mu[0]*N - mu[1]*S - beta*I*S/N
# dI/dt = alpha*E - (mu[1]+gamma)*I
# dR/dt = gamam*I - mu[1]*R

#   where:
#    beta = average number of contacts per day
#    gamma = average duration of disease (in day)

def derivSIR(state, t, N, mu, beta, gamma):
    S, I, R = state
    dSdt = mu[0]*N - mu[1]*S - beta*I*S/N
    dIdt = beta*I*S/N - (mu[1]+gamma)*I
    dRdt = gamma*I - mu[1]*R
    return dSdt, dIdt, dRdt


def deriv_modifiedSIR(state, t, N, mu):
    beta = 0.5
    gamma = 1/7.0
    S, I, R = state
    dSdt = mu[0]*N - mu[1]*S - beta*I*S/N
    if (I/N < 0.0001):
        beta = 0.5
    else:
        beta = 0.2
    dIdt = beta*I*S/N - (mu[1]+gamma)*I
    dRdt = gamma*I - mu[1]*R
    return dSdt, dIdt, dRdt
## Dynamical equations (SEIR model):

# dS/dt = mu[0]*N - mu[1]*S - beta*I*S/N - kappa*E*S/N
# dE/dt = beta*I*S/N + kappa*E*S/N - (mu[1]+alpha)*E
# dI/dt = alpha*E - (mu[1]+gamma)*I
# dR/dt = gamam*I - mu[1]*R

def derivSEIR(state, t, N, mu, alpha, beta, gamma, kappa):
    S, E, I, R = state
    dSdt = mu[0]*N - mu[1]*S - beta*I*S/N - kappa*E*S/N
    dEdt = beta*I*S/N + kappa*E*S/N - (mu[1]+alpha)*E
    dIdt = alpha*E - (mu[1]+gamma)*I
    dRdt = gamma*I - mu[1]*R
    return dSdt, dEdt, dIdt, dRdt
# initialize:
N = 27000000    # population of Wuhan
E0 = 1          # number of initially exposed people

model = 'SIR'
num_days = 450
t = np.arange(0,num_days,0.1)
# birth and death rates
mu = [0.0,0.0] # assume pop. constant in this scenario

print("Assume a population of {:,} people.".format(N))

## transition rates between stages of the disease
alpha = 1/14.0      # 14 day incubation period
gamma = 1/5.0       # 7 day infectious period before death/recovery
 
print("Assume the virus lasts (on average) in an infectious person for {:.2f} days, once infectious.".format(1.0/gamma))

## transmission rates = average number of contacts per person * probability of transmission
# incubation period
av_contacts = 3  
prob_of_trans = 0.1
kappa = av_contacts*prob_of_trans

# infectious period
av_contacts = 2
prob_of_trans = 0.125
beta = av_contacts*prob_of_trans

# solve the ODE system
if model == 'SEIR':
    arguments = (N, mu, alpha, beta, gamma, kappa)
    S, E, I, R = N-E0, E0, 0, 0
    intial_state = S, E, I, R
    func = derivSEIR
elif model == 'SIR':
    arguments = (N, mu, beta, gamma)
    S, I, R = N-E0, E0, 0
    intial_state = S, I, R
    func = derivSIR
    print("Beta value of {} contacts per person per day.".format(beta))
    R0 = beta/gamma
    print("R0 value would then be {:.2f} (=beta/gamma)".format(R0))
elif model == 'modifiedSIR':
    arguments = (N, mu)
    S, I, R = N-E0, E0, 0
    intial_state = S, I, R
    func = deriv_modifiedSIR

# solve the ODE
ret = odeint(func, intial_state, t, args=arguments)

if model == 'SEIR':
    total_exposed = E+I
else:
    S, I, R = ret.T
    total_exposed = I


# print some summary information
print("Total who got the virus: {:,}".format(np.round(R[-1],0)))
print(".... as a percentage: {:.2f}%".format(100*R[-1]/N))
print("Total who never got the virus: {:,}".format(np.round(S[-1],0)))
print(".... as a percentage: {:.2f}%".format(100*S[-1]/N))
print("Range of the number of deaths (1-7% of cases): {:,} to {:,}.".format(np.round(R[-1]*0.01,0),np.round(R[-1]*0.07,0)))
print("Assuming a mortality rate of 4%, the number of deaths:  {:,}.".format(int(R[-1]*0.04)))
print("Peak number of simultaneously infected people: {:,}".format(np.round(np.max(total_exposed),0)))
print("Approximate timing of the peak: {:.2f}".format(np.round(t[np.argmax(total_exposed)],0)))
# plotting functions
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/N, 'g', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, total_exposed/N, 'y', alpha=0.5, lw=2, label='Total exposed')
ax.plot(t, R/N, 'b', alpha=0.5, lw=2, label='Recovered')
if model=='SIR':
    title_string = "Spread of coronavirus COVID-19, R0={}".format(np.round(R0,3))
else:
    title_string = "Spread of coronavirus COVID-19"
plt.title(title_string)
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