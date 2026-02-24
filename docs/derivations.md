# Mathematical Derivations

## 1. Problem Formulation

### 1.1 Multi-Asset Dynamics with Regime Switching

We model $d$ risky assets and one risk-free asset. The risky asset prices follow a regime-dependent geometric Brownian motion:

$$dS_t^i = S_t^i \left[ \mu_i(\alpha_t) \, dt + \sigma_i(\alpha_t) \, dW_t^i \right], \quad i = 1, \ldots, d$$

where $\alpha_t$ is a continuous-time Markov chain on $\{1, \ldots, K\}$ with generator matrix $Q = (q_{kl})$ satisfying:
- $q_{kl} \geq 0$ for $k \neq l$
- $\sum_l q_{kl} = 0$ for all $k$

The Brownian motions $W^i$ have correlation structure $\langle dW^i, dW^j \rangle = \rho_{ij} \, dt$.

The risk-free asset evolves as $dB_t = r B_t \, dt$.

### 1.2 Wealth Dynamics

Let $\pi_t^i$ denote the fraction of wealth invested in risky asset $i$. The wealth process evolves as:

$$dW_t = W_t \left[ r + \sum_{i=1}^d \pi_t^i (\mu_i - r) \right] dt + W_t \sum_{i=1}^d \pi_t^i \sigma_i \, dW_t^i - W_t \sum_{i=1}^d \varepsilon |\delta \pi_t^i| \, d\Lambda_t$$

where $\varepsilon$ is the proportional transaction cost rate and $d\Lambda_t$ is the rebalancing measure.

### 1.3 Objective

Maximise CRRA expected utility of terminal wealth:

$$V(W, p, t) = \sup_{\pi} \mathbb{E}\left[ \frac{W_T^\gamma}{\gamma} \,\middle|\, W_t = W, \pi_t = p \right]$$

where $\gamma < 1$, $\gamma \neq 0$ is the risk aversion parameter.

---

## 2. Wonham Filter

### 2.1 Continuous-Time Formulation

Since the regime $\alpha_t$ is not directly observed, we maintain the filter:

$$p_t^k = \mathbb{P}(\alpha_t = k \mid \mathcal{F}_t^S), \quad k = 1, \ldots, K$$

The Wonham filter SDE is:

$$dp_t^k = \sum_l q_{lk} p_t^l \, dt + p_t^k \sum_{i=1}^d \frac{\mu_i(k) - \bar{\mu}_i(p_t)}{\sigma_i(\alpha_t)} \left( \frac{dS_t^i}{S_t^i} - \bar{\mu}_i(p_t) \, dt \right)$$

where $\bar{\mu}_i(p) = \sum_k p^k \mu_i(k)$ is the posterior-averaged drift.

### 2.2 Discrete-Time Approximation

We discretise using Euler's method with time step $\Delta t$:

$$p_{n+1}^k = p_n^k + \sum_l q_{lk} p_n^l \,\Delta t + p_n^k \sum_{i=1}^d \frac{\mu_i(k) - \bar{\mu}_i}{\sigma_i^2} \left( \Delta \log S^i - \bar{\mu}_i \,\Delta t \right)$$

followed by projection onto the probability simplex.

---

## 3. HJB Equation

### 3.1 Standard HJB

The value function satisfies the Hamilton–Jacobi–Bellman equation:

$$V_t + \sup_\pi \mathcal{H}(\pi, V) = 0$$

where the Hamiltonian is:

$$\mathcal{H}(\pi, V) = W \left[ r + \pi^T(\mu - r\mathbf{1}) \right] V_W + \frac{1}{2} W^2 \pi^T \Sigma \pi \, V_{WW} + \text{belief dynamics terms}$$

with $\Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}$.

The optimal (frictionless) Merton weight is:

$$\pi^*_i = \frac{\mu_i - r}{(1 - \gamma) \sigma_i^2}$$

### 3.2 Transaction Cost Modification

With proportional costs $\varepsilon$, the HJB becomes a variational inequality with three regions:
- **Buy region**: optimal to increase $\pi$
- **Sell region**: optimal to decrease $\pi$
- **No-trade region**: optimal to hold

The no-trade boundaries depend on the local curvature of $V$ and the cost $\varepsilon$.

### 3.3 Belief Dynamics in HJB

The belief state $p$ enters as an additional state variable:

$$V_t + \sup_\pi \left\{ \mathcal{H} + \sum_{k,l} q_{lk} p^l \frac{\partial V}{\partial p^k} + \frac{1}{2} \sum_{k,l} \beta_{kl} \frac{\partial^2 V}{\partial p^k \partial p^l} \right\} = 0$$

---

## 4. Hansen–Sargent Robust Control

### 4.1 Multiplier Preferences

The investor is uncertain about the true model. Under Hansen–Sargent preferences, the investor solves:

$$V(W, p, t) = \sup_\pi \inf_h \mathbb{E}\left[ \frac{W_T^\gamma}{\gamma} + \int_t^T \frac{\theta}{2} |h_s|^2 \, ds \right]$$

where $h_t$ is the drift distortion under the alternative measure, and $\theta > 0$ controls the level of ambiguity aversion:
- **Small $\theta$**: more ambiguity averse (large distortions are cheap)
- **Large $\theta$**: less ambiguity averse (converges to standard model)

### 4.2 Worst-Case Distortion

The inner minimisation over $h$ yields:

$$h_t^* = -\frac{\sigma \cdot \pi \cdot V_W}{\theta}$$

### 4.3 Robust HJB

Substituting back, the robust HJB is:

$$V_t + \sup_\pi \left\{ \mathcal{H}(\pi, V) - \frac{1}{2\theta} (\sigma \cdot \pi)^2 V_W^2 \right\} = 0$$

The robust Merton weight is:

$$\pi_{\text{robust}}^i = \frac{\mu_i - r}{(1 - \gamma)\sigma_i^2 + \sigma_i^2 V_W / (\theta V)}$$

showing a **shrinkage** of the allocation relative to the standard Merton solution.

---

## 5. Numerical Methods

### 5.1 Finite Difference Discretisation

State space: $x = \log W$ (log-wealth) and $p$ (belief). We use:
- **Sinh-stretched** non-uniform grid in $x$ for resolution near $x = 0$
- **Uniform** grid in $p \in [\epsilon, 1-\epsilon]$

Operators:
- Central differences for first derivative $\partial_x$ (second-order)
- Standard 3-point stencil for $\partial_{xx}$ (second-order)
- Upwind differences for belief dynamics (stability)

### 5.2 IMEX Time Stepping

We use an Implicit-Explicit (IMEX) scheme:
- **Implicit**: diffusion term $V_{xx}$ (unconditionally stable)
- **Explicit**: Hamiltonian optimisation (allows nonlinear max over $\pi$)

At each time step $t_n = T - n \Delta t$:

$$\frac{V^{n+1} - V^n}{\Delta t} + D_{xx} V^{n+1} = -\sup_\pi \mathcal{H}_{\text{explicit}}(\pi, V^n)$$

### 5.3 Boundary Conditions

- **Wealth boundaries**: Neumann (zero-flux) conditions $\partial_x V = 0$ at $x_{\min}$ and $x_{\max}$
- **Belief boundaries**: Neumann conditions $\partial_p V = 0$ at $p = 0$ and $p = 1$

### 5.4 Terminal Condition

$$V(W, p, T) = \frac{W^\gamma}{\gamma}$$

---

## 6. Deep RL Formulation

### 6.1 MDP Structure

- **State**: $(W_t, \pi_t^{\text{current}}, p_t, \tau_t)$ — wealth, current weights, belief, time remaining
- **Action**: target portfolio weights $\tilde\pi_t$ on the simplex
- **Reward**: CRRA terminal utility $U(W_T) = W_T^\gamma / \gamma$
- **Transition**: GBM with regime switching + Wonham filter update

### 6.2 PPO Algorithm

We use Proximal Policy Optimization with:
- **Actor**: outputs Dirichlet distribution over portfolio simplex (natural constraint satisfaction)
- **Critic**: scalar value estimate
- **GAE**: Generalized Advantage Estimation with $\lambda = 0.95$
- **Clipping**: $\epsilon = 0.2$

### 6.3 Scalability Advantage

| Dimension $d$ | PDE Grid $O(N^{d+1})$ | RL Params $O(d)$ |
|---|---|---|
| 2 | $N^3$ | ~50K |
| 5 | $N^6$ | ~55K |
| 10 | $N^{11}$ | ~60K |

The PDE approach suffers from the **curse of dimensionality**: computational cost scales exponentially with the number of assets. Deep RL scales polynomially.

---

## 7. EM Calibration (Baum–Welch)

### 7.1 Model

Observed log-returns $\Delta \log S_t$ are modelled as emissions from a Hidden Markov Model with Gaussian emissions.

### 7.2 Algorithm

1. **E-step**: Forward-backward to compute $\gamma_t(k) = P(\alpha_t = k | \text{data})$
2. **M-step**: Update $\hat\mu_k$, $\hat\Sigma_k$, and transition matrix $P_{kl}$
3. **Iterate** until convergence

### 7.3 Generator Estimation

From the discrete transition matrix $P = \exp(Q \Delta t)$, recover $Q$ via matrix logarithm:

$$Q = \frac{1}{\Delta t} \log(P)$$

---

## References

1. Merton, R.C. (1969). *Lifetime Portfolio Selection under Uncertainty*. Review of Economics and Statistics.
2. Merton, R.C. (1971). *Optimum Consumption and Portfolio Rules in a Continuous-Time Model*. Journal of Economic Theory.
3. Wonham, W.M. (1965). *Some Applications of Stochastic Differential Equations to Optimal Nonlinear Filtering*. SIAM J. Control.
4. Hansen, L.P. & Sargent, T.J. (2001). *Robust Control and Model Uncertainty*. American Economic Review.
5. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
6. Davis, M.H.A. & Norman, A.R. (1990). *Portfolio Selection with Transaction Costs*. Mathematics of Operations Research.
