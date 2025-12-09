module SemiringConfig

export SemiringConfigOptions, build_semiring

struct SemiringConfigOptions
    name::Symbol
end

function build_semiring(config::SemiringConfigOptions)
    if config.name == :tropical
        return (:tropical, :min_plus)
    elseif config.name == :boolean
        return (:boolean, :logic)
    else
        error("Unknown semiring $(config.name)")
    end
end

end # module
