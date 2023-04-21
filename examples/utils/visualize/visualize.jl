# using Makie, Colors
using WGLMakie
using Colors

includet("utils.jl")
includet("axis_setup.jl")
includet("components.jl")

### Top-level functions for making different types of figures ###

plot_obs(t_to_obs, domain; model_label="", occluder=nothing, kwargs...) = 
    plot_single_panel(
        domain;
        obs=t_to_obs,
        model_label, occluder, kwargs...
    )

function plot_obs(t_to_obs, domain; n_backtrack=1)
    f, ax, t = setup_ax(domain)
    plot_obs!(ax, t, t_to_obs; n_backtrack)
    return (f, t)
end

plot_obs_and_distribution(
    t_to_obs, t_to_distribution, domain, label;
    title=nothing,
    maxtime_for_colorrange=nothing,
    occluder=nothing, kwargs...
) = plot_single_panel(domain;
    obs=t_to_obs, heatmaps=t_to_distribution,
    model_label=label, occluder, maxtime_for_colorrange, title, kwargs...
)

plot_obs_particlescurrentpos_and_distribution(
    t_to_obs, t_to_distribution, t_to_particles,
    particle_filter_description, domain, label;
    title=nothing,
    maxtime_for_colorrange=nothing,
    occluder=nothing, kwargs...
) = plot_single_panel(domain;
    obs=t_to_obs, heatmaps=t_to_distribution,
    model_label=label, occluder, maxtime_for_colorrange, title,
    particles=t_to_particles, particle_filter_description, kwargs...
)

plot_observation_and_particles(domain, t_to_obs, t_to_particles) =
    plot_obs_particlestrajectories(
        t -> (t_to_obs(t),), nothing,
        t -> [
            (logweight, [(position,) for position in trajectory])
            for (logweight, trajectory) in t_to_particles(t)
        ],
        "", domain, ""
    )

plot_obs_particlestrajectories(
    t_to_obs, t_to_distribution, t_to_particles,
    particle_filter_description, domain, label;
    title=nothing,
    maxtime_for_colorrange=nothing,
    occluder=nothing, kwargs...
) = plot_single_panel(domain;
    obs=t_to_obs,
    model_label=label, occluder, maxtime_for_colorrange, title,
    particles=t_to_particles, particle_filter_description,
    plot_particles=((args...,) -> [plot_particles_trajectories!(args...), plot_particles_currentpos!(args...; do_jitter=false)]),
    heatmaps=t_to_distribution, kwargs...
)

plot_multiple_particle_filter_results(domain, inference_labels_and_t_to_particles; kwargs...) =
    plot_multiple_inference_grids(
        domain, inference_labels_and_t_to_particles;
        plot_particles=plot_particles_trajectories_and_currentpoints!,
        kwargs...
    )

plot_trajectory(domain, t_to_position) = plot_trajectory_and_observations(domain, t_to_position, nothing)
function plot_trajectory_and_observations(domain, t_to_position, t_to_observation)
    return plot_obs_particlestrajectories(
        isnothing(t_to_observation) ? (t -> ()) : (t -> (t_to_observation(t),)),
        nothing,
        t -> collect(zip(
            [1],
            [ # particle idx
                [ # time idx
                    ( # object idx
                        t_to_position(k),
                        )
                        
                    for k=1:t]
                    
                ]
        )),    
        "",
        domain,
        ""
    );
end

### Functions for single-panel and multi-panel figure construction ###

#=
- obs, heatmaps, and particles should be arrays indexed by time
- obs gives the observed blips
- heatmaps gives, at each timestep, a heatmap over positions at that timestep
- particles[t] should be a trajectory
=#
function plot_single_panel(
    domain;
    obs=nothing, heatmaps=nothing, particles=nothing,
    model_label="", title=nothing,
    maxtime_for_colorrange=nothing,
    particle_filter_description = "",
    occluder=nothing,
    plot_particles=plot_particles_currentpos!,
    slider_maxval = nothing,
    n_backtrack=100
)
    f, ax, t = setup_ax(domain)

    (hm, occ, obsplt, particle_plt) = plot_onto_single_panel!(ax, t, heatmaps, occluder, obs, particles, plot_particles, maxtime_for_colorrange; n_backtrack)

    if !isnothing(hm)
        Colorbar(f[1, 2], hm; vertical = true)
    end

    f[2, :] = t_label = Label(f, lift(t -> "t = $t", t))

    # l = make_legend(f[3, 1], (heatmaps, occluder, particles, obs), (hm, occ, obsplt, particle_plt), particle_filter_description)

    # f[4, :] = model_label_element = Label(f, model_label; justification=:left)

    for element in (t_label,)
        element.tellheight = true
        element.tellwidth = false
    end

    if !isnothing(slider_maxval)
        s = Slider(f[5, :], range=1:slider_maxval)
        connect!(t, s.value)
    end

    return (f, t)
end

function plot_multiple_inference_grids(
    domain, inference_labels_and_t_to_particles;
    plot_particles,
    obs=nothing, heatmaps=nothing,
    model_label="", title=nothing,
    maxtime_for_colorrange=nothing,
    particle_filter_description = "",
    occluder=nothing,
    show_obs_only_axis=false,
    slider_maxval=nothing
)
    # TODO: adjust figure size!
    n_axes = length(inference_labels_and_t_to_particles) + (show_obs_only_axis ? 1 : 0)
    f = Figure(resolution=(300 + 600 * n_axes, 1000))
    t = Observable(1)
    axes_layout = f[1, 1] = GridLayout()
    i = 1

    if show_obs_only_axis
        ax = axes_layout[1, i] = Axis(f, title="Observed Blips", aspect=DataAspect())
        setup_ax!(ax, domain)
        plot_onto_single_panel!(ax, t, nothing, occluder, obs, nothing, plot_particles, maxtime_for_colorrange)
        i += 1
    end

    (occ, hm, obsplt, particle_plt) = nothing, nothing, nothing, nothing
    for (label, t_to_particles) in inference_labels_and_t_to_particles
        ax = axes_layout[1, i] = Axis(f, title=label, aspect=DataAspect())
        setup_ax!(ax, domain)
        (hm, occ, obsplt, particle_plt) = plot_onto_single_panel!(ax, t, heatmaps, occluder, obs, t_to_particles, plot_particles, maxtime_for_colorrange)
        i += 1
    end

    c = axes_layout[1, i] = Colorbar(axes_layout[1, i], hm; vertical=true)

    f[2, :] = t_label = Label(f, lift(t -> "t = $t", t))
    # l = make_legend(f[3, 1], (heatmaps, occluder, first(inference_labels_and_t_to_particles)[2], obs), (hm, occ, obsplt, particle_plt), "inference")

    # f[4, :] = model_label_element = Label(f, model_label; justification=:left)

    for element in (t_label)
        element.tellheight = true
        element.tellwidth = false
    end

    if !isnothing(slider_maxval)
        s = Slider(f[5, :], range=1:slider_maxval)
        connect!(t, s.value)
    end

    return (f, t)
end

### Construct individual panels + legend 

function plot_onto_single_panel!(
    ax, t, heatmaps, occluder, obs, particles, plot_particles, maxtime_for_colorrange;
    n_backtrack=100
)
    hm =
        if isnothing(heatmaps)
            nothing
        else
            dist_heatmap!(ax, t, heatmaps, maxtime_for_colorrange)
        end
    occ =
        if isnothing(occluder)
            nothing
        else
            plot_occluder!(ax, occluder)
        end
    obsplt =
        if isnothing(obs)
            nothing
        else
            plot_obs!(ax, t, obs; n_backtrack)
        end
    particle_plt =
        if isnothing(particles)
            nothing
        else
            plot_particles(ax, t, particles)
        end

    return (hm, occ, obsplt, particle_plt)
end
truncate_obs_plt(obsplt) = obsplt[1:min(length(obsplt), 3)]
obsplt_labels(obsplt) =
    if length(obsplt) ≤ 3
        [
            "Blips at time $(i == 0 ? "t" : "t - $i")"
            for i=0:(length(obsplt)-1)
        ]
    else
        ["Blips at time t", "Blips at time t-1", "..."]
    end
    
make_legend(figpos, (heatmaps, occluder, particles, obs), (hm, occ, obsplt, particle_plt), particle_filter_description) =
    Legend(
        figpos,
        maybe_array([
            (heatmaps, PolyElement(color=:gray)),
            (occluder, occ),
            (particles, particle_plt),
            (obs, obsplt, truncate_obs_plt, true)
        ]),
        maybe_array([
            (heatmaps, "P(Square occupied at t | blips_{1:t})  [from exact inference]"),
            (occluder, "Occluder"),
            (particles, "Particles from $particle_filter_description"),
            (obs, obsplt, obsplt_labels, true)
        ]),
    )

### Video Utils ###

record!(f, filename, t, maxtime; framerate=1, mintime=1) =
    record(f, filename, mintime:maxtime; framerate) do _t
        t[] = _t
    end

function animate!(t, maxtime; framerate=30, starttime=1)
    @async for i=starttime:maxtime
        t[] = i
        sleep(1/framerate)
    end
end

### Misc
simple_logweight_plot(logweights) = 
    plot([Point2f(i, logweight) for (i, lws) in enumerate(logweights) for logweight in lws]; axis=(xlabel="Time", ylabel="log(SMC weight)"))

simple_weight_plot(logweights) = 
    plot([Point2f(i, exp(logweight)) for (i, lws) in enumerate(logweights) for logweight in lws]; axis=(xlabel="Time", ylabel="SMC weight"))