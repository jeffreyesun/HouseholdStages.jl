# Field-graph walk backing the relocation-completeness checks. The stopping rule mirrors
# `to_device`'s own: a closure, a layout and a spec are boundaries the relocation does not cross,
# so the arrays behind them (a lowered reward source's captured axis grid, a layout's node vector,
# a spec's constant transition) are host-resident by design and are not collected.
#
# The checks lean on a coincidence: for every type in the tree today, "the walker stops here"
# (`Function` / `AbstractLayout` / `AbstractStageSpec`) and "the relocation stops here" (no
# `adapt_structure` method, so Adapt's fallback carries the object through) pick out the same
# objects. A plain struct that legitimately stays host-resident without being one of those three
# would be walked into and reported as an incomplete relocation — a false positive, not a bug in
# `to_device`. Extend the stopping rule alongside the relocation rules.

"""
Every array reachable from `x` through the relocation graph, in visit order.
"""
function reachable_arrays(x)
    out  = Any[]
    seen = Base.IdSet{Any}()
    _walk!(out, seen, x)
    return out
end

function _walk!(out, seen, x)
    (x isa Function || x isa Type || x isa Symbol) && return out
    x isa AbstractLayout && return out
    x isa AbstractStageSpec && return out
    isbitstype(typeof(x)) && return out
    x in seen && return out
    push!(seen, x)
    if x isa AbstractArray
        push!(out, x)
    elseif x isa Base.RefValue
        _walk!(out, seen, x[])
    elseif x isa Union{Tuple, NamedTuple}
        for v in x; _walk!(out, seen, v); end
    elseif isstructtype(typeof(x))
        for i in 1:nfields(x); _walk!(out, seen, getfield(x, i)); end
    end
    return out
end
