_unpack_maybearray_shorthand(tup) =
    if length(tup) == 4
        tup
    elseif length(tup) == 3
        (tup[1], nothing, (_,) -> tup[2], tup[3])
    else
        @assert length(tup) == 2
        (tup[1], nothing, ((_,) -> tup[2]), false)
    end

"""
Accepts an iterator over values of the form
`(key, function_input, value_function, do_unroll)`.

Returns an array containing, for each `key` which is is not nothing,
- `value_function(function_input)` if `!do_unroll`
- `value_function(function_input)...` if `do_unroll`

In the input array,
`(key, value)` may be used as a shorthand for `(key, _, (_,) -> value, false)`,
and `(key, value, do_unroll)` may be used as a shorthand for
`(key, _, (_,) -> value, do_unroll)`.
"""
maybe_array(arr) =
    (
        (
            if isnothing(key)
                ()
            else
                val = value_function(input)
                if do_unroll
                    val
                else
                    (val,)
                end
            end
        )
        for (
            key, input, value_function, do_unroll
        ) in Iterators.map(_unpack_maybearray_shorthand, arr)
    ) |> Iterators.flatten |> collect