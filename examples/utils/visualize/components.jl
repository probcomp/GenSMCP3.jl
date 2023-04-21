MIN_VIS_WEIGHT() = 0.05
TRAJ_WIDTH() = 3

normalize(v) = exp.(v .- logsumexp(v))

plot_obs!(ax, t, t_to_obs; n_backtrack=1) =
    plot_temporal_points!(ax, t, t_to_obs ; n_backtrack, color=colorant"red")

function plot_temporal_points!(ax, t, time_to_pts;
    n_backtrack=0, color, markersize=45, marker=:circle
)
    plts = []
    # Current point:
    pt = @lift(Point2[Point2(pt) for pt in time_to_pts($t)])
    scatter!(ax,
        pt; color, markersize, marker
    ) |> plt -> push!(plts, plt)
    
    # previous points
    for i=1:n_backtrack
        scatter!(ax,
            @lift($t - i < 1 ? Point2[] : Point2[Point2(pt) for pt in time_to_pts($t - i)]);
            color=RGBA(color, max(0.3, 0.75 * 0.9^(i - 1))),
            markersize = markersize * max(0.05, 0.75 * 0.9^(i-1)),
            marker
        ) |> plt -> push!(plts, plt)
    end

    return plts
end

# time_to_particles(t) is a list of `(logweight, [(position₁¹, ..., position₁ⁿ), ..., (positionₜ¹, ..., positionₜⁿ)])` tuples
# where `n` is the number of objects
function plot_particles_currentpos!(ax, t, _time_to_particles;
    max_markersize=45, marker=:circle, do_jitter=true
)
    # time t to vector of tuples (logweight, (positionₜ¹, ..., positionₜⁿ)) for each particle
    time_to_particles(t) = [
        (logweight, last(traj)) for (logweight, traj) in _time_to_particles(t)
    ]
    n_particles = t -> length(time_to_particles(t))
    time_to_grouped_points = t -> [points for (logweight, points) in time_to_particles(t)]
    time_to_points = t -> time_to_grouped_points(t) |> Iterators.flatten |> collect
    time_to_jittered_points = !do_jitter ? time_to_points : (t -> 
        [
            (x + rand(Normal(0, 0.1)), y + rand(Normal(0, 0.1)))
            for (x, y) in time_to_points(t)
        ])

    time_to_weights_flat = 
        t -> exp.([logweight for (logweight, _) in time_to_particles(t)])
    time_to_relative_weights_flat = t ->
        let ws = time_to_weights_flat(t)
            ws./maximum(ws)
        end
    time_to_weight_dist_flat = t -> begin
        weights = time_to_weights_flat(t)
        if isempty(weights)
            weights
        else
            normalize(weights)
        end
    end
    time_to_unflattened_weights(time_to_flat_weights) =
        t -> (
            (w for obj in points)
            for (w, points) in zip(time_to_flat_weights(t), time_to_grouped_points(t))
        ) |> Iterators.flatten |> collect


    # Doubling the marker size doubles the marker radius.
    # We want a particle with twice as much weight to take up twice as much _area_.
    # So we want the markersize proportional to the square root of the weight.
    return scatter!(ax,
        @lift(map(Point2, time_to_jittered_points($t)));
        markersize = @lift(max_markersize * sqrt.(time_to_unflattened_weights(time_to_weight_dist_flat)($t))),
        marker,
        color=RGBA(colorant"navy", 0.5)
        # color=@lift(
        #     [
        #         RGBA(colorant"navy", alpha) for alpha in
        #         [
        #             min(a, 1.0)
        #             for a in time_to_unflattened_weights(time_to_weight_dist_flat)($t) * n_particles($t)/10
        #         ]
        #     ]
        # )
    )
end
# each t_to_particles(t) is an array of elements (logweight, trajectory)
function unpack_object_trajectories(multiobject_trajectory)
    # multiobject_trajectory looks like [(position₁¹, ..., position₁ⁿ), ..., (positionₜ¹, ..., positionₜⁿ)]
    n_objects = length(first(multiobject_trajectory))
    @assert all(length(objs) == n_objects for objs in multiobject_trajectory)
    object_trajectories = [
        [state[i] for state in multiobject_trajectory]
        for i=1:n_objects
    ]
    return object_trajectories
end
# Do this to make sure that even 0-weight trajectories 
function plot_particles_trajectories!(ax, t, t_to_particles)
    logweights = @lift([lw for (lw, traj) in t_to_particles($t)])
    trajectories = @lift([traj for (lw, traj) in t_to_particles($t)])
    
    _plot_trajectories!(ax, logweights, trajectories, colorant"navy")
end
plot_particles_trajectories_and_currentpoints!(args...,) =
    [
        plot_particles_trajectories!(args...),
        plot_particles_currentpos!(args...; do_jitter=false)
    ]
function _plot_trajectories!(
    ax, unfiltered_logweights_observable, unflitered_trajectories_observable,
    base_color=colorant"navy"; # only used if `colors` is not provided
    colors=nothing
)
    filter_indices = lift(
        unfiltered_logweights_observable ->
            begin
                v = [i for (i, lw) in enumerate(unfiltered_logweights_observable) if !(isinf(lw) || isnan(lw))]
                v
            end,
            unfiltered_logweights_observable
    )
    logweights_observable = @lift($unfiltered_logweights_observable[$filter_indices])
    trajectories_observable = @lift($unflitered_trajectories_observable[$filter_indices])

    line_data = (
        lift(trajectories_observable) do trajectories
            x = collect(Point2, Iterators.flatten(
                # Iterators.flatten(vcat(map(Point2, trajectory), [NaN])
                vcat(map(Point2 ∘ only, trajectory), [Point2(NaN, NaN)])
                for trajectory in trajectories
            ))
            x
        end
    )
    if isnothing(colors)
        colors = (
            lift(logweights_observable) do logweights
                if isempty(logweights)
                    []
                else
                    normalized_lws = logweights .- logsumexp(logweights)
                    [
                        let w = exp(lw)
                            if isnan(w) || isinf(w) || isapprox(0, w)
                                RGBA(base_color, 0.)
                            elseif w < MIN_VIS_WEIGHT()
                                RGBA(base_color, MIN_VIS_WEIGHT())
                            else
                                RGBA(base_color, w)
                            end
                        end
                        for lw in normalized_lws
                    ]
                end
            end
        )
    end
    color_vector = (
        lift(trajectories_observable, colors) do trajs, cols
            x = (collect ∘ Iterators.flatten)(
                (color for _=1:(length(trajectory) + 1))
                for (color, trajectory) in zip(cols, trajs)
            )
            x
        end
    )
    lines!(ax, line_data; color=color_vector, linewidth=TRAJ_WIDTH())
end

# Version which uses `t` directly to help avoid errors due to observable update order.
function __plot_trajectories!(
    ax, t, t_to_unfiltered_logweights, t_to_unflitered_trajectories,
    base_color=colorant"navy"; # only used if `colors` is not provided
    colors=nothing
)
    filter_indices = (
        t ->
            begin
                v = [i for (i, lw) in enumerate(t_to_unfiltered_logweights(t)) if !(isinf(lw) || isnan(lw))]
                v
            end
    )
    logweights_observable = @lift(t_to_unfiltered_logweights($t)[filter_indices($t)])
    trajectories_observable = @lift(t_to_unflitered_trajectories($t)[filter_indices($t)])

    line_data = (
        lift(trajectories_observable) do trajectories
            x = collect(Point2, Iterators.flatten(
                # Iterators.flatten(vcat(map(Point2, trajectory), [NaN])
                vcat(map(Point2f ∘ only, trajectory), [Point2(NaN, NaN)])
                for trajectory in trajectories
            ))
            x
        end
    )

    if isnothing(colors)
        colors = (
            lift(logweights_observable) do logweights
                if isempty(logweights)
                    []
                else
                    normalized_lws = logweights .- logsumexp(logweights)
                    [
                        let w = exp(lw)
                            if isnan(w) || isinf(w) || isapprox(0, w)
                                RGBA(base_color, 0.)
                            elseif w < MIN_VIS_WEIGHT()
                                RGBA(base_color, MIN_VIS_WEIGHT())
                            else
                                RGBA(base_color, w)
                            end
                        end
                        for lw in normalized_lws
                    ]
                end
            end
        )
    end
    color_vector = (
        lift(trajectories_observable, colors) do trajs, cols
            x = (collect ∘ Iterators.flatten)(
                (color for _=1:(length(trajectory) + 1))
                for (color, trajectory) in zip(cols, trajs)
            )
            x
        end
    )
    lines!(ax, line_data; color=color_vector, linewidth=TRAJ_WIDTH())
end

plot_occluder!(ax, occluder) =
    poly!(
        [Rect(x - 0.5, y - 0.5, 1., 1.) for (x, y) in occluder],
        strokecolor=colorant"gold",
        strokewidth=2,
        color=RGBA(0,0,0,0)
    )
maybe_plot_occluder!(ax, o) = isnothing(o) ? nothing : plot_occluder!(ax, o)

function dist_heatmap!(ax, t, t_to_distribution, maxtime_for_colorrange=nothing)
    colormap = cgrad([:white, :lightgray, :gray], [0, 1e-4, 1.0], scale=exp)
    colorrange =
        if isnothing(maxtime_for_colorrange)
            @lift (0, min(maximum(t_to_distribution($t))*1.05, 1.0))
        else
            overall_max_prob = maximum(maximum(t_to_distribution(_t)) for _t=1:maxtime_for_colorrange)
            (0, min(overall_max_prob, 1.0))
        end
    hm = heatmap!(
        ax, @lift(t_to_distribution($t));
        colormap, colorrange
    )
    return hm
end

plot_occluder!(ax, occluder::Rect2) =
    poly!(
        Rect(occluder.x - 0.5, occluder.y - 0.5, occluder.w + 1, occluder.h + 1),
        strokecolor=colorant"gold",
        strokewidth=2,
        color=RGBA(0,0,0,0)
    )