# PPO at undergraduate level

We use **standard** Stable-Baselines3 PPO with an MLP policy (64×64).
Nothing about the algorithm is new; the project studies its *application*
to MLFQ parameter tuning.

## 1. Policy and value functions

The policy $\pi_\theta(a|s)$ is a neural network that outputs a probability
over the 8 MLFQ configurations for each observed state. The value function
$V(s)$ (a second head/network) estimates how much future reward to expect
from state $s$. The policy *acts*; the value function *judges*.

## 2. Expected cumulative reward

$$J(\pi)=\mathbb E_\pi\left[\sum_t\gamma^t r_t\right]$$

Read it as: "follow policy $\pi$, add up rewards, discounting later rewards
by $\gamma^t$, and average over everything random (arrivals, I/O, action
sampling)." Training means increasing $J$.

## 3. Policy gradients

$$\nabla_\theta J(\theta)=
\mathbb E_\pi[\nabla_\theta\log\pi_\theta(a_t|s_t)A_t]$$

Intuition: nudge the network to make actions with positive advantage $A_t$
more likely, and actions with negative advantage less likely. $\log\pi$
appears because differentiating a probability's log turns "how likely was
this action" into a well-scaled push.

## 4. Advantage estimation

$A_t \approx$ (reward actually obtained) − (reward the value function
expected). $A_t > 0$ means "better than expected — do this more."
SB3 uses GAE (Generalized Advantage Estimation, $\lambda=0.95$), which
smooths this signal over several steps to reduce variance.

## 5. PPO clipping

Vanilla policy gradients can take a destructive giant step. PPO limits how
far the new policy may move from the old one via the probability ratio
$r_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$:

$$L^{CLIP}(\theta)=
\mathbb E_t[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$$

with $\epsilon = 0.2$. If an update would change an action's probability by
more than ±20%, the clipped term stops rewarding further movement. Result:
many small, safe improvements instead of rare catastrophic ones — which is
why PPO is stable enough to train on a laptop for this project.

## Hyperparameters (configurable in `learnsched.train.DEFAULT_HPARAMS`)

learning rate 3e-4, rollout 512 steps, batch 64, 10 epochs, $\gamma=0.99$,
GAE $\lambda=0.95$, clip 0.2, entropy bonus 0.01, MLP [64, 64], fixed seeds.
