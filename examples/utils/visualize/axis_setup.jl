function setup_ax!(ax, domain)
    dx, dy = domain
    xlims = (first(dx) - 0.5, last(dx) + 0.5)
    ylims = (first(dy) - 0.5, last(dy) + 0.5)
    xlims!(ax, xlims)
    ylims!(ax, ylims)

    ax.xticks = floor.(LinRange(first(dx), last(dx), 4))
    ax.yticks = floor.(LinRange(first(dy), last(dy), 4))
    ax.xminorticks=first(dx):last(dx)
    ax.yminorticks=first(dy):last(dy)
    ax.xminorticksvisible = true
    ax.yminorticksvisible = true
    ax.xgridvisible = false
    ax.ygridvisible = false
end

function setup_ax(domain; kwargs...)
    resolution =
        if haskey(kwargs, :resolution)
            kwargs[:resolution]
        else
            dx, dy = domain
            dx[2] - dx[1] < 15 ? (700, 800) : (900, 1000)
        end
    f = Figure(; kwargs..., resolution)
    ax = Axis(f[1, 1]; aspect=DataAspect())
    t = Observable(2)
    setup_ax!(ax, domain)

    return (f, ax, t)
end