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

#=
Stratified SMCP3 update.

The `strata` argument is an iterator of `Gen.ChoiceMap`s, one for each stratum.
Each choicemap constrains addresses in the generative model.
The SMCP3 K kernel will be given each stratum as an additional (final) argument.
(So the SMCP3 update passed into this should have the K_args be 1 element too short.)
=#
function GenParticleFilters.pf_update!(
    state::Gen.ParticleFilterState,
    new_args::Tuple,
    argdiffs::Tuple,
    observations::Gen.ChoiceMap,
    strata,
    update::SMCP3Update
)
    n_particles = length(state.traces)
    n_strata = length(strata)
    @assert n_particles >= n_strata
    GenParticleFilters.stratified_map!(n_particles, strata) do i, stratum
        state.new_traces[i], log_weight_update = GenTraceKernelDSL.run_smcp3_step(
            state.traces[i], new_args, argdiffs,
            merge(observations, stratum),
            update.K, update.L,
            (update.K_args..., stratum),
            update.L_args;
            check_are_inverses=update.check_are_inverses
        )
        state.log_weights[i] += log_weight_update + log(n_strata)
    end
    GenParticleFilters.update_refs!(state)
    return state
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
