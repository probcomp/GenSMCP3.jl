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
    update::SMCP3Update,
    observations::Gen.ChoiceMap
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

export @kernel, SMCP3Update

end # module GenSMCP3
