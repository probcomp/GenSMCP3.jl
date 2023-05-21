# GenSMCP3.jl
Automated Sequential Monte Carlo with Probabilistic Program Proposals (SMCP<sup>3</sup>), for [Gen](https://gen.dev).

SMCP<sup>3</sup> is a family of Sequential Monte Carlo algorithms in which particle-updating proposal distributions can be any member of a very broad class of probabilistic programs.  Given probabilistic programs representing the proposal distributions, and the probabilistic model targetted by inference, SMCP<sup>3</sup> automatically generates a particle filter that can be used to perform inference in the targetted model.  This repository contains a Julia implementation of automated SMCP<sup>3</sup>, for use with the [Gen](https://gen.dev) probabilistic programming language.

For details about SMCP<sup>3</sup>, please see our paper:
```
SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals
Alex Lew*, George Matheos*, Tan Zhi-Xuan, Matin Ghavamizadeh, Nishad Gothoskar, Stuart Russell, Vikash Mansinghka
AISTATS 2023
```
For the code used to generate the results and figures in the paper, see [this repository](https://github.com/probcomp/aistats2023-smcp3).

## Installation
GenSMCP3 is implemented in [Julia](https://julialang.org/).

GenSMCP3 interoperates with the [Gen](https://gen.dev) probabilistic programming language, and the [GenParticleFilters](https://github.com/probcomp/GenParticleFilters.jl) module of Gen, which adds sequential Monte Carlo (particle filtering) functionality to Gen.

To install GenSMCP3, first install Gen and ParticleFilters, then install GenSMCP3.  From the Julia REPL:
```julia
] add Gen
] add GenParticleFilters
] add https://github.com/probcomp/DynamicForwardDiff.jl
] add https://github.com/probcomp/GenTraceKernelDSL.jl
] add https://github.com/probcomp/GenSMCP3.jl
```

## Tutorial
For an in-depth tutorial, see the [the SMCP<sup>3</sup> tutorial notebook](examples/notebooks/Tutorial.ipynb).

## Basic Example and components of an SMCP<sup>3</sup> algorithm

### Example probabilistic model
To give a simple usage example, consider the following probabilistic model with an underlying latent state `x`.
At each timestep `t`, a new noisy observation of `x` is received.
```julia
using Gen
@gen function model(t)
    # `x` is the latent value to be inferred
    x ~ normal(0, 100)
    
    # a number of noisy observations are made of `x`.
    # the goal will be to infer P(x | observations).
    # whenever a new observation is made, we will update this posterior
    # using a particle filter.
    observations = []
    for i in 1:t
        obs = {"obs$i"} ~ normal(x, 1)
        push!(observations, obs)
    end

    return observations
end
```

We can generate random samples from the model.  The will look like this:
```julia
# Simulate a random trace from the model, with 3 observations of x.
trace = Gen.simulate(model, (3,))
display(get_choices(random_trace))
# │
# ├── "obs3" : 244.0100682859606
# │
# ├── "obs2" : 244.82494671853553
# │
# ├── "obs1" : 244.85189759030885
# │
# └── :x : 245.56601014895355
```
A `trace` is a data structure that contains all of the random choices made by the model probabilistic program.  In code snippet, the trace contains the values of `x` and the three observations of `x`.  Each is choice in the trace is associated with an address (in the above, `:x` and `"obs1"`, `"obs2"`, `"obs3"` respectively).

With this model in hand, we can now tackle the inference problem in which we want to infer the posterior distribution `P(x | observations)`.

### Forward proposal probabilistic program

A SMCP<sup>3</sup> algorithm is defined by two probabilistic program proposals.  The first proposal distribution, the "forward proposal", is responsible for updating hypothetical latent states, or "particles", in light of new data.  The second proposal distribution, the "backward proposal", is responsible for _inverting_ the forward proposal.  (This will be explained in the next section.)

The forward proposal distribution is run at each timestep, to update the particles in light of new data.  It accepts as input a "particle" `previous_trace` from time t-1, which is a trace of the model probabilistic program called with the argument `t-1`.  (That is, it will be a trace of the model containing a value of `x`, and `t-1` observations of `x`.)  It must output two things:
1. A specification for how to update `previous_trace` into a new trace, which will be the new particle.  It is through this output that the proposal distribution specifies how to update the particle.  In particular, this specification will be a _choicemap_ `new_latents` that specifies a new value for every latent variable in the model.  A new trace will be produced by updating `previous_trace` to (1) now accept `t` as an argument, rather than `t-1`, (2) overwrite all the latent values with those specified in `new_latents`, and (3) add a new observation of `x` to the trace (this is mangaged by the code which runs the particle filter, not in the proposal distribution).
2. A specification of the random choices that the backward proposal distribution would make to invert this particular particle update.  This is a choicemap `backward_choices` containing a value for every random choice the backward proposal would make.

```julia
# Import the `@kernel` DSL used to write probabilistic program proposals.
import GenSMCP3: @kernel

# This probabilistic program defines a forward proposal distribution for the model above.
# It will receive as input a trace of the model with t-1 observed datapoints,
# and a new observation.  Its job is to propose an update to the latent state
# of the model, to incorporate information in the new observation.
@kernel function forward_proposal(previous_trace, new_observation)
    t_prev = get_args(previous_trace)[1]
    t = t_prev + 1

    # Construct a vector of all the observations, as of time t.
    old_observations = [previous_trace["obs$i"] for i in 1:t_prev]
    new_observations = vcat(old_observations, [new_observation])

    # Compute the mean and variance of the new observations.
    mean = sum(new_observations) / t
    var = 1/(t + 1)

    # Propose a new value for x, based on the new observations.

    std = sqrt(var)
    new_x ~ normal(mean, std)

    # Return two things.
    # First: return a choicemap which tells Gen how to overwrite the latent state
    # of the current trace, to produce the new trace proposed by this update.
    # This is the return value which specifies the updated latent state.
    # 
    # Second: return a choicemap containing all of the choices the _backward proposal_ (defined below)
    # would make to invert this update.  (In this case, the backward proposal would have to propose
    # what the value of x was before the update, in order to invert it.)
    return (
        choicemap((:x, new_x)),
        choicemap((:previous_x, trace[:x]))
    )
end
```

### Backward proposal probabilistic program

SMCP<sup>3</sup> algorithms require a _backward proposal distribution_ which approximately inverts the forward proposal distribution.  This is a probabilistic program that takes as input a trace of the model with `t` observed datapoints.  Its job is to propose:
1. What may the hypothesized latent state of the model have been, before the forward proposal was applied with the newest observation?  (This will be specified as a choicemap `previous_latents` that specifies a value for every latent variable in the model.  The proposed previous trace is obtained from the updated trace by (1) now accepting `t-1` as an argument, rather than `t`, (2) overwriting all the latent values with those specified in `previous_latents`, and (3) removing the last observation of `x` from the trace.)
2. What random choices may the forward proposal distribution have made, in order to update the latent state in light of the new data?  This will be a choicemap containing a value for every random choice the forward proposal would make.

```julia
@kernel function backward_proposal(updated_trace, new_observation)
    t = get_args(updated_trace)[1]
    
    if t > 1
        observations_before_update = [updated_trace["obs$i"] for i in 1:t-1]

        mean = sum(observations_before_update) / (t-1)
        var = 1/t
        std = sqrt(var)
    else
        # If t=1, the previous x value was generated at random from the prior,
        # without taking into account any observations.  So to revert to
        # this old latent state, we'll sample the previous x from normal(0, 1).
        mean = 0
        std = 1
    end

    # Propose what the value of x was before the update.
    previous_x ~ normal(mean, std)

    # Return two things.
    # First: return a choicemap which tells Gen how to overwrite the latent state to invert the update
    # from the forward proposal.
    # Second: return a choicemap containing all of the choices the _forward proposal_ (defined above)
    # would make to re-apply this update.  (In this case, the forward proposal would have to propose
    # what the value of x was after the update, in order to re-apply it.)
    return (
        choicemap((:x, previous_x)),
        choicemap((:new_x, updated_trace[:x]))
    )
end
```

### Running Sequential Monte Carlo
Using these proposal distributions, we can define a full sequential Monte Carlo algorithm which uses SMCP<sup>3</sup> to update the particles.  We use the `pf_initialize` and `pf_update!` methods from the library [GenParticleFilters](https://github.com/probcomp/GenParticleFilters.jl) to define the sequential Monte Carlo algorithm.  `pf_initialize` initializes a particle filter by initializing `n_particles` latent hypotheses ("particles").  Then, whenever a new observed value becomes available, the `pf_update!` method is used to update the particles in light of the new observation.  We use an `SMCP3Update` to specify that we want to use SMCP<sup>3</sup> to update the particles.  The `SMCP3Update` constructor takes as arguments the forward and backward proposal distributions we defined above.

```julia
import GenSMCP3: SMCP3Update
using GenParticleFilters

function run_smcp3(observations, n_particles)
    # Initialize a particle filter (SMC algorithm).
    # Pass in an empty choicemap to indicate that at time 0, there are not
    # yet any observed values.
    state = pf_initialize(model, (0,), choicemap(), n_particles)

    # For each tth observation:
    for (t, observation) in enumerate(observations)
        pf_update!(
            state,
            # new argument to the model, to have it output `t` observations
            (t,),

            # Tell Gen some change occurred to the argument, but we are
            # not going to provide any special information about the
            # type of change.  (In some cases we may provide information
            # about the change, so Gen can use incremental computation
            # to improve performance.)
            (UnknownChange(),),
            
            # update the latent state using an SMCP<sup>3</sup> update,
            # with the forward and backward proposals defined above,
            # given the new observation as an additional argument
            # after the trace to be updated
            SMCP3Update(
                forward_proposal,
                backward_proposal,
                (observation,),
                (observation,)
            ),
            
            # update the trace to have this observation
            # at the address "obs$t"
            choicemap(("obs$t", observation))
        )
    end
    
    # Resample the particles whenever the ESS becomes too small.
    # (This will duplicate high-weight particles, and prune low-weight particles.)
    if effective_sample_size(state) < 1/5 * n_particles
        # Perform residual resampling, pruning low-weight particles
        pf_resample!(state, :residual)
    end
    
    return state
end
```

We can now run inference.  Here I show the inference results, first on a stream of 3 observed values (1, 2, 3), and then on a stream of 4 observed values (1, 2, 3, 10).
```julia
inference_result_state = run_smcp3([1, 2, 3], 1000);
empirical_expected_x = mean(inference_result_state, :x) # = 2.002

inference_result_state_2 = run_smcp3([1, 2, 3, 10], 1000);
empirical_expected_x_2 = mean(inference_result_state, :x) # = 4.0002
```

Note that in more sophisticated examples, the code could be written so that new observations are streamed into the particle filter online, rather than being passed in at the start in a vector.

See [this notebook](examples/notebooks/Very_Simple_Example.ipynb) to run this example.  See [this notebook](examples/notebooks/Tutorial.ipynb) for a tutorial on SMCP<sup>3</sup>, using a simple online state estimation problem as an example (rather than this trivial problem of estimating a single value).

## Writing probabilistic program proposals
GenSMCP3 exposes a new DSL for writing probabilistic programs which are to be used as proposal distributions.  Probabilistic programs are written in this DSL using Julia functions preceded by the `@kernel` macro.

(A different DSL than Gen's regular probabilistic-program DSL is necessary due to limitations in Gen's current support for automatic-differentiation through probabilistic programs.  Improvements to this support are under-way in in-developments variants of Gen, such as [GenJax](https://github.com/probcomp/genjax).)

### The `@kernel` DSL
Probabilistic program proposals are written in the `@kernel` DSL.  (It is called the `@kernel` DSL because it is used to write proposals, which are [Markov Kernels](https://en.wikipedia.org/wiki/Markov_kernel).)  A probabilistic program written in this DSL (which we will call a kernel) is written as a Julia function preceded by the `@kernel` macro.  It may accept arguments, and should return a value.  (In SMCP3, the return value should be a pair of choicemaps: one specifying how to update the model trace, and one constraining the choices of the backward/forward proposal.)  The first argument value to a kernel will always be a trace of the model probabilistic program, to which the kernel is supposed to propose an update.

A kernel's body may contain deterministic Julia code, as well as `~` expressions,
familiar from Gen:

* `{:x} ~ dist(args)` samples from a Gen distribution at address `:x`
* `{:x} ~ gen_fn(args)` samples from a Gen generative function at address `:x`, **and evaluates to the _trace_ of the function, rather than its return value**
* `{:x} ~ kernel_fn(args)` calls another `@kernel`-defined function at address `:x`, **and evaluates to its return value.**

As in Gen, `x = {:x} ~ f()` can be shortened to `x ~ f()`, and—for generative function or kernel calls—the `{*} ~ f()` syntax can be used to splice the choices made by `f` into the "top level" of the caller's choicemap.

No stochasticity should be invoked in a kernel, except through `~` expressions.

## Implementation of automated SMCP<sup>3</sup>, and related libraries
This implementation of automated SMCP<sup>3</sup> is implemented in this repository, and the following two repositories we developed in the process of developing SMCP<sup>3</sup>:
1. [GenTraceKernelDSL](https://github.com/probcomp/GenTraceKernelDSL.jl): this introduces the DSL for writing probabilistic programs which are to be used as proposal distributions.  (The `@kernel` macro is a re-export from this library.)
2. [DynamicForwardDiff](https://github.com/probcomp/DynamicForwardDiff.jl): this is an implementation of automatically sparsity-aware forward-mode automatic differentiation.  It is used to compute change-of-measures correction terms needed for the implementation of SMCP<sup>3</sup>.  In SMCP3, this correction is the absolute value of the determinant Jacobian of the function which maps the inputs to a probabilistic program proposal, and the collection of random choices of that probabilistic program, to the output of the probabilistic program proposal.  (See Theorem 2 of our paper for details.)

## Citation
Please cite
```
@InProceedings{smcp3,
  title = 	 {SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals},
  author =   {Lew, Alexander K. and Matheos, George and Zhi-Xuan, Tan and Ghavamizadeh, Matin and Gothoskar, Nishad and Russell, Stuart and Mansinghka, Vikash K.},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {7061--7088},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/lew23a/lew23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/lew23a.html},
  abstract = 	 {This paper introduces SMCP3, a method for automatically implementing custom sequential Monte Carlo samplers for inference in probabilistic programs. Unlike particle filters and resample-move SMC (Gilks and Berzuini, 2001), SMCP3 algorithms can improve the quality of samples and weights using pairs of Markov proposal kernels that are also specified by probabilistic programs. Unlike Del Moral et al. (2006b), these proposals can themselves be complex probabilistic computations that generate auxiliary variables, apply deterministic transformations, and lack tractable marginal densities. This paper also contributes an efficient implementation in Gen that eliminates the need to manually derive incremental importance weights. SMCP3 thus simultaneously expands the design space that can be explored by SMC practitioners and reduces the implementation effort. SMCP3 is illustrated using applications to 3D object tracking, state-space modeling, and data clustering, showing that SMCP3 methods can simultaneously improve the quality and reduce the cost of marginal likelihood estimation and posterior inference.}
}
```