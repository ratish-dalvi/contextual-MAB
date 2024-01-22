# Contextual MAB
A solution for contextual multi-armed-bandit problem with constraints, using linear programming


## Introduction

Multi-armed bandits (MAB) is a classic formulation of the exploration versus exploitation problem - 
It is a hypothetical experiment where a bandit must choose between multiple slot machines,
each with an unknown payout. More background on the problem can be found here: [1](https://en.wikipedia.org/wiki/Multi-armed_bandit), [2](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)

Contextual MAB is an extension of this approach where we factor in the bandit’s environment, or context, when choosing an action. For the scope of this problem, we are going to reduce the contextual bandit problem into a supervised learning problem, where an "oracle" takes into account the context and gives us a score that indicates the probability of success on the next action.



## Matching Patients to Providers

One real-world application of contextual MAB problem in healthcare is sending patient referrals to a network of mental health providers (MHPs). Let's consider a patient referral system that matches a new patient with the "best" available MHP. The system does not have access to real-time availability of the MHP, but it knows the number of new patients the MHP can generally take week over week. This is the maximum number of referrals a MHP can have (1). Similarly, the MHP expects a minimum number of referrals every week, to keep them engaged on the system, and that sets a minimum constraint on the number of referrals (2). 
When a patient is considered for a provider, there is a lot of information that could be used to make a successful match. Patients have conditions that they seek treatments for, and they might have preferences on gender, race, age, specialization etc of the provider. Similarly, the system has data on the provider's history in declining and scheduling referrals. This information containing patient's preferences and provider's history, constitutes the "context" for the problem (3). 
All three of these together make this use-case equivalent to a contextual MAB problem with minimum and maximum constraints:
- A patient referral can be thought of as "playing" a machine; similarly, MHPs are slot machines. 
- The payout of a slot machine is equivalent to the inherent "scheduling rate" of a MHP. Scheduling rate of a MHP is the proportion of the received referrals scheduled for an appointment.
- The inherent scheduling rate varies over time, but we will assume that it remains nearly the same if we manage to honor the maximum and minimum constraints. A MHP will remain happy if they get a minimum number of referrals each week, and not more than the maximum they want. If we don't honor the constraints, the MHP might become disengaged and stop accepting the referrals
- New providers are on-boarded on the system often, which makes exploration a consistent part of the system. 
- Even though MHPs have an "inherent" scheduling rate, it is common for their rate to vary over time due to staffing changes, seasonal variation, business changes etc. 

Unlike the goal of a typical MAB problem, where the goal is to lower "regret", or the difference between an optimal solution and your solution, this system's goal is to optimize the total number of appointments scheduled by the system, across all MHPs.

Referrals are sent out in real-time, which means we cannot do a batch assignment, and that makes this problem tricky.

## Satisfying Constraints with a Perfect Oracle

For the scope of this solution, we are only going to focus on the constraint satisfaction problem. For simplicity, we assume an oracle that accurately predicts the likelihood of success if a patient is sent to a MHP. This oracle, in practice, is a well-calibrated supervised learning algorithm that predicts the odds that a referral will get scheduled if we send it to a given MHP. 

This formulation violates the standard MAB problem– where the focus is on minimizing regret. One might argue that this is not a classic MAB problem anymore;
Before implementing this solution, we used the classic solution for the MAB problem of Thompson sampling [1](https://en.wikipedia.org/wiki/Thompson_sampling) that minimizes regret. However, we realized that minimizing regret is not an issue at all. On the contrary, we found that greedy algorithms that usually have higher regret bounds or no bound at all do just as well as Thompson sampling for this use-case, because recently onboarded MHPs are eager to schedule referrals; but as time passes, failure to maintain constraints leads to their dissatisfaction and a drop in their "payout" aka scheduling rate. Also, Thompson sampling does not allow us to put constraints on the MHPs. The oracle, which is a supervised learning algorithm, uses context for new MHPs to predict their success rate quite well, and it reduced our concern for exploration and regret.

So, this problem can be simplified into the following:
- A system consists of sending m referrals to n MHPs
- An oracle gives likelihood for each referral-MHP pair
– MHPs have an inherent/true scheduling rate
- Each MHP has a maximum and minimum constraint on the number of referrals they can receive

The simplest solution to this problem could be to use a greedy algorithm for sending a referral to a MHP. For any given referral, we will send it to the MHP with the highest payout, but then, there is no
guarantee that it will satisfy the minimum and maximum constraints. Let's consider an example to understand the problem: 

#### Example:

Let's consider a provider MHP1 which is nearing its maximum quota/cap, but is not full yet. Let's say we can only send them two more referrals this week, but they end up being the top pick for four referrals. How do we decide which two to send?

Let's say the referrals, oracle scores, and MHPs look like this:
    
   |    |mhp1 | mhp2 | mhp3|
   |----|-----|------|-----|
   | r1 | 0.7 | 0.65 | 0.2 |
   | r2 | 0.8 | 0.7  | 0.75|
   | r3 | 0.9 | 0.2  | 0.1 |
   | r4 | 0.7 | 0.24 | 0.3 |

<b>Strategy 1 </b>

We suspend the MHP after the quota is met - after the first two referrals in this case. 
That means: 
- r1 and r2 are sent to MHP1 and 
- r3 is sent to MHP2 
- r4 is sent to MHP3
    
What are the expected number of appointments in this case? Assuming that the scores from the oracle are true probabilities.<br/>
Expected number of appointments = 0.7 + 0.8 + 0.2 + 0.3 = 2

<b>Strategy 2</b>

- r3 and r4 appointments to mhp1 instead
- r1 to mhp2
- r2 to mhp3
   
Expected number of appointments = 0.65 + 0.75 + 0.9 + 0.7 = 3 <br/>
So, a slightly different strategy resulted in one extra appointment in this group. This shows that a greedy approach might not be the best approach here.

In this case, we can easily see which two referrals to route to MHP1 because we can see all the referrals. Unfortunately, in production, the referrals are sent in real-time, as they come, and we have no information about the future. So, when deciding for r1, we do not know about r2, r3 and r4. We still want to try and optimize the total number of appointments. But how?

### Linear Optimization without Assignment
It is clear that we have reduced this to a standard linear optimization problem: it has an objective function (expected number of appointments), a cost function (scores) and constraints to satisfy. But there is one issue - we cannot send referrals in batches. They are sent in real-time as soon as a patient is received by the system.

This means that we cannot do "assignment" like a typical linear optimization problem. This notebook describes a solution that using linear optimization for this problem without assignment by using score adjustments 


### Formulation

#### Classic Binary Optimization

Let's assume we could batch referrals, what would the solution look like?

Let's say there are N MHPs, R referrals, this algorithm gives a probability score for each referral

Maximum volume: $q_{0}, q_{1}, q_{2} ... q_{N}$

Minimum volume: $ p_{0}, p_{1}  ... p_{N}$ 

Let's consider the score matrix 



   |     |mhp1| mhp2|mhp3| ... |
   |---- |-----|------|-----|-----|
   | p0  | 0.7 | 0.64 | 0.1 | ... |
   | p1  | 0.8 | 0.58  | 0.32| ... |
   | p2  | 0.91 | 0.35 | 0.3 | ... |
   | ... | ... | ...  | ... | ... |


For example, $s_{01} = 0.64$ , $s_{22} = 0.35$ 


Assignment Matrix:

   |     |mhp1| mhp2|mhp3| ... |
   |---- |-----|------|-----|-----|
   | p1  |  0  |   1  |  0  | ... |
   | p2  |  0  |   0  |  1  | ... |
   | p3  |  1  |   0  |  0  | ... |
   | p4  |  1  |   0  |  0  | ... |
   | ... | ... | ...  | ... | ... |
   
<b>Formulation </b>:
    
$$ \max_{a} \sum \sum a_{ij} s_{ij}$$

    
$$ \sum_i a_{ij} < d_j  \qquad \forall  j $$

$$ \sum_i a_{ij} > c_j  \qquad \forall  j $$
    
$$ \sum_j a_{ij} = 1  \qquad \forall  i $$
    

So far so good, but this formulation is not very useful because we cannot send referrals to providers in batches. In the absence of assignments, we use compute score adjustments, which are listed below: 


#### Linear optimization w/o assignment

We force assignment to be on the maximum score per referral, which makes it a purely endogenous variable - as opposed to a decision variable. <br/>
We introduce a new set of decision variables $P$ (for score adjustments) such that $P_j$ is the score adjustment (boost + or penalty -) for each MHP $M_j$


   | j  |---> |      |     |     |
   |----|-----|------|-----|-----|
   | p  | 0.7 | 0.64 | 0.1 | ... |

<b>Formulation </b>:

$$ \max_{a} \sum \sum a_{ij} s_{ij}$$

    
$$ \sum_i a_{ij} < d_j  \qquad \forall  j $$
    
$$ \sum_i a_{ij} > c_j  \qquad \forall  j $$

$$ \sum_j a_{ij} = 1  \qquad \forall  i $$

$$ s_{ij} + r_j \leq \sum_k a_{ik} (s_{ik} + r_k)  \qquad \forall  i \forall  j $$


There is a problem with the last equation. <br/>

It is not linear!


#### Convert Quadratic to Linear Using a Trick: McCormick Envelopes

$ z = xy $ 
where x is binary and y is continuous

$ L \leq y \leq U $ 
where L and U are bounds on y

Then the following constraints will make it linear

$$ z \leq Ux$$
$$ z \geq Lx$$
$$ z \leq y - L(1-x) $$
$$ z \geq y - U(1-x) $$


#### Final Formulation


$$ \max_{a} \sum \sum a_{ij} s_{ij} - (\sum abs (p_{j})) $$

    
$$ \sum_i a_{ij} < d_j  \qquad \forall  j $$

$$ \sum_i a_{ij} > c_j  \qquad \forall  j $$

$$ \sum_j a_{ij} = 1  \qquad \forall  i $$

$$ s_{ij} + p_j \leq \sum_k a_{ik} s_{ik} + \sum_k z_{ik}   \qquad \forall  i  \forall  j$$


$$ z_{ik} = a_{ik} p_k $$

$$ -a_{ik} \leq z_{ik} \leq a_{ik} $$

$$ r_k - (1 - a_{ik}) \leq z_{ik} \leq r_k + (1 - a_{ik}) $$


That's it! <br/> Refer to the notebook for the implementation.

