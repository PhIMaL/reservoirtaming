# Chaotic systems prediction using structured reservoir computing and model discovery

The main idea of this project is to combine Jonathan's work on [structured reservoir computing](https://arxiv.org/pdf/2006.07310.pdf) to predict a chaotic system (in this case, the Kuramoto-Sivashinsky equation.) with DeepMoD. The hope is that if we combine some regularization in the form of a physical equation, the system should be able to predict further. In short, combining the benefits of a purely data-driven method with that of actual modeling. In this document I (very) shortly wrote down some possible directions, milestones and some other but more experimental directions. 

## Directions

I see two main directions; one focuses on inductive biases of the reservoir (i.e. work on the high dimensional stuff) and another focuses on improving the mapping to the observale (i.e. work on the low dimensional stuff (mostly)).

**Inductive biases for the echo state network**

Reservoir computing now uses a propagation of the hidden state;
$$
x^{t+1} = f(Wx^{t} + b)
$$
While this works pretty well, there doesn't seem to be any conservation or anything in there. I propose to combine reservoir computing with [hamiltonian neural networks](https://arxiv.org/pdf/1906.01563.pdf), which conserve energy. Since we're applying this to the reservoir, we're implicitly assuming a Hamiltonian exists for the high-dimensional reservoir - a pretty big assumption, but it might just work out. Roughly speaking, a hamiltonian network would do this:
$$
x^{t+1} = x^{t} + \partial \mathcal{H}\Delta t
$$
where the hamiltonian $\mathcal{H}$ would be modeled by the reservoir:
$$
\mathcal{H} = f(Wx^{t} + b)
$$
A very nice testcase could be the chaotic pendulum!

**Improving the order reduction**

We could make a more direct improvement by making the only trainable layer, the output layer, physically consistent; right now, the output layer is simply a ridge regression. Someone has already done this for [PINNs](https://arxiv.org/abs/2001.02982), so let's start exploring this way with the deepmod framework. We could consider bayesian methods, since the KS equation is chaotic and thus very sensitive to the actual value of the coefficients. 

## Milestones

Gert-Jan will try to finish milestones 1-3 (possibly with some help from jonathan) so priyanka doesn't have to start from scratch.

1. Write reservoir network in Jax 

   Implement a working version of reservoir networks in Jax, simply using dense layers.

2. Improve KS simulation code

   The KS code seems a bit messy, let's clean it up and move it to Jax.

3. Implement structured transform in Jax

   We use structured reservoir computing, so we need to implement the fast hadamard-walsh transform. The pytorch library is old and doesn't seem maintained. Using Jax and XLA should make it much faster without having to write cuda.

4. Direction 2 - Deepmod echo-pinns

   Pinns have already been implemented (see references), so a good first attempt would be simply to use echo state networks as interpolator for deepmod. Main thing to figure out is how to use automatic differentiation efficiently to calculate the derivatives. Since we test on chaotic systems, uncertainty in coefficients has a massive effect. It might be beneficial to explore some bayesian methods since the coefficient is a distribution, rather than a single number.

5. Direction 1 - Hamiltonian

   Once we have a basic version going, an interesting question is how inductive biases help the whole thing. Hamiltonian NNs seem fairly straight forward to implement, so this is a good second step.

6. Direction 2 - Unknown equation

   With DeepMoD we assume the equation is in our set of possible terms - but what if it isn't? Can we use a neural network to model the unknown equation (Schaeffer et al. showed that giving the features $u_xx$ etc to a NN helps prediction)? One direction I've been meaning to explore is using attention: if we use an attention model on the inputs, can we discover the underlying terms indirectly and without defining a library?

## Random ideas and questions

Below I'm writing some random ideas and questions I had after reading the paper, which might or might not be worth digging into.

* Depending on the hyperparameters, the reservoir may be unstable. I found a [paper](https://www.nature.com/articles/s41598-019-50158-4.pdf) which self-normalizes the activation and claims to get rid of these issues. If true, could make our life a lot easier.

- The reservoir network in Jonathan's paper is single depth. If you'd use a very deep model (with shared weights), the output would converge to some fixed value. This is an idea explored with [Deep Equilibrium Models](http://implicit-layers-tutorial.org/). Instead of placing many layers together, you simply use a root finding technique to find the fixed point. What would make it easier for us is that we don't have to backpropagate so it's a very simple implementation. You basically get infinitely deep echo state networks!
- Echo state networks use simple regularized regression to map the reservoir space to the observable space. This greatly simplifies training, but if we see the last layer as a dimensionality reduction operation, we can use many more cool things: why not use Principal Component Analysis (PCA), Koopman Analysis or dynamic mode decomposition?

