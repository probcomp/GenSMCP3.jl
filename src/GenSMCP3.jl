#=
TODO: document the pf initialization methods, in GenTraceKernelDSL.jl and here.
TODO: add an example of pf initialization to the readme.
TODO: remove the method of pf_update! with the SMCP3Update before the observations,
    and update the readme and examples.
=#

module GenSMCP3

import Gen, GenParticleFilters, GenTraceKernelDSL
using GenTraceKernelDSL: Kernel, @kernel

struct SMCP3Update
    K                  :: Kernel
    L                  :: Kernel
    K_args             :: Tuple
    L_args             :: Tuple
    check_are_inverses :: Bool
end
SMCP3Update(K, L, K_args, L_args) = SMCP3Update(K, L, K_args, L_args, false)

# In the release of GenSMCP3, I introduced this method of pf_update!.
# However, it now looks to me like it's more consistent with GenParticleFilters
# to have the update after the observations.  So:
# TODO: remove this method, and update the readme and examples.
GenParticleFilters.pf_update!(
    state::Gen.ParticleFilterState,
    new_args::Tuple,
    argdiffs::Tuple,
    update::SMCP3Update,
    observations::Gen.ChoiceMap,
) = GenParticleFilters.pf_update!(
    state, new_args, argdiffs, observations, update
)

function GenParticleFilters.pf_update!(
    state::Gen.ParticleFilterState,
    new_args::Tuple,
    argdiffs::Tuple,
    observations::Gen.ChoiceMap,
    update::SMCP3Update
)
    n_particles = length(state.traces)
    for i=1:n_particles
        state.new_traces[i], log_weight_update = GenTraceKernelDSL.run_smcp3_step(
            state.traces[i], new_args, argdiffs,
            observations,
            update.K, update.L, update.K_args, update.L_args;
            check_are_inverses=update.check_are_inverses
        )
        state.log_weights[i] += log_weight_update
    end
    GenParticleFilters.update_refs!(state)
end

function GenParticleFilters.pf_initialize(
    model::Gen.GenerativeFunction{T,U}, model_args::Tuple,
    observations::Gen.ChoiceMap, n_particles::Int, update::SMCP3Update;
    dynamic::Bool=false
) where {T,U}
    V = dynamic ? Gen.Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (traces[i], log_weights[i]) = GenTraceKernelDSL.run_initial_smcp3_step(
            model, model_args, observations,
            update.K, update.L, update.K_args, update.L_args;
            check_are_inverses=update.check_are_inverses
        )
    end
    return Gen.ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

export @kernel, SMCP3Update

end # module GenSMCP3
