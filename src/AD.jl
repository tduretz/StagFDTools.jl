using DifferentiationInterface

const AUTO_DIFF_BACKEND = AutoForwardDiff()
export ad_jacobian_first, gradient, gradient!, prepare_gradient, prepare_jacobian, jacobian!, Constant, AUTO_DIFF_BACKEND

struct ConstArg{T}
    value::T
end

struct DuplicatedArg{T, G}
    value::T
    grad::G
end

Const(x) = ConstArg(x)
Duplicated(x, grad) = DuplicatedArg(x, grad)

@inline _ad_contexts(args...) = map(Constant, args)
@inline _ad_promote_arg(x) = x
@inline _ad_promote_arg(x::StaticArrays.MArray) = SArray(x)

@inline ad_gradient(f, x, args...) = gradient(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)
@inline ad_value_and_gradient(f, x, args...) = value_and_gradient(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)
@inline ad_derivative(f, x, args...) = derivative(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)
@inline ad_value_and_derivative(f, x, args...) = value_and_derivative(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)
@inline ad_jacobian(f, x, args...) = jacobian(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)
@inline ad_value_and_jacobian(f, x, args...) = value_and_jacobian(f, AUTO_DIFF_BACKEND, x, _ad_contexts(args...)...)

@inline function _replace_tuple_entry(xs::Tuple, idx::Int, x)
    return ntuple(i -> i == idx ? x : xs[i], length(xs))
end

function ad_partial_gradients(f::F, xs::Tuple, args...) where {F}
    promoted_xs = map(_ad_promote_arg, xs)
    return ntuple(i -> gradient(x -> f(_replace_tuple_entry(promoted_xs, i, x)..., args...), AUTO_DIFF_BACKEND, promoted_xs[i]), length(xs))
end

function ad_value_and_jacobian_first(f::F, x, args...) where {F}
    full_value = f(x, args...)
    first_value, jac = value_and_jacobian(z -> first(f(z, args...)), AUTO_DIFF_BACKEND, x)
    return first_value, jac
end

function ad_jacobian_first(f::F, x, args...) where {F}
    first_value, jac = value_and_jacobian(z -> first(f(z, args...)), AUTO_DIFF_BACKEND, x)
    return first_value, jac
end

ad_unwrap(x) = _ad_promote_arg(x)
ad_unwrap(x::ConstArg) = _ad_promote_arg(x.value)
ad_unwrap(x::DuplicatedArg) = _ad_promote_arg(x.value)

struct ForwardDiffResult{V, D}
    val::V
    derivs::Tuple{D}
end

Base.getindex(res::ForwardDiffResult, i::Int) = res.derivs[i]

forwarddiff_normalize_val(value) = value isa Number ? (value,) : value
forwarddiff_primary_output(value) = value isa Tuple ? first(value) : value

replace_arg(args::Tuple, idx::Int, value) = ntuple(i -> i == idx ? value : args[i], length(args))

function forwarddiff_gradients!(f, args...)
    values = map(ad_unwrap, args)
    for (idx, arg) in pairs(args)
        if arg isa DuplicatedArg
            grad = gradient(x -> f(replace_arg(values, idx, x)...), AUTO_DIFF_BACKEND, values[idx])
            arg.grad .= grad
        end
    end
    return nothing
end

function forwarddiff_gradient(f, x, args...)
    values = map(ad_unwrap, args)
    val, grad = value_and_gradient(f, AUTO_DIFF_BACKEND, x, map(Constant, values)...)
    return ForwardDiffResult(forwarddiff_normalize_val(val), (grad,))
end

function forwarddiff_jacobian(f, x, args...)
    values = map(ad_unwrap, args)
    full_val = f(x, values...)
    prim = x isa Number ? value_and_derivative(z -> forwarddiff_primary_output(f(z, values...)), AUTO_DIFF_BACKEND, x) :
        value_and_jacobian(z -> forwarddiff_primary_output(f(z, values...)), AUTO_DIFF_BACKEND, x)

    _, deriv = prim
    wrapped_deriv = if full_val isa Tuple && !(x isa Number)
        ntuple(i -> (deriv[:, i],), length(x))
    else
        deriv
    end

    return ForwardDiffResult(forwarddiff_normalize_val(full_val), (wrapped_deriv,))
end
