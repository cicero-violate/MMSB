module ErrorRecovery

using ..FFIWrapper: LIBMMSB
using ..RustErrors: RustFFIError
export RetryPolicy, exponential_backoff, retry_with_backoff

struct RetryPolicy
    max_attempts::Int
    base_delay_ms::Int
    max_delay_ms::Int
    backoff_factor::Float64
end

RetryPolicy(;max_attempts=3,base_delay_ms=100,max_delay_ms=5000,backoff_factor=2.0)=RetryPolicy(max_attempts,base_delay_ms,max_delay_ms,backoff_factor)
exponential_backoff(a,p) =min(round(Int,p.base_delay_ms*(p.backoff_factor^(a-1))),p.max_delay_ms)
is_retryable_error(e)    =(e isa RustFFIError ? ccall((:mmsb_error_is_retryable,LIBMMSB),Bool,(Int32,),e.code) : e isa IOError)
is_fatal_error(e)        =(e isa RustFFIError ? ccall((:mmsb_error_is_fatal,LIBMMSB),Bool,(Int32,),e.code) : false)

function retry_with_backoff(f,p=RetryPolicy())
    for a in 1:p.max_attempts
        try;return f();catch e;is_fatal_error(e)&&rethrow(e);!is_retryable_error(e)&&rethrow(e);a==p.max_attempts&&rethrow(e);sleep(exponential_backoff(a,p)/1000);end
    end
end

end
