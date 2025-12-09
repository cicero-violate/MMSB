"""
    StructuralPreferences

Structural preferences for system organization.
"""
module StructuralPreferences

export Preference, evaluate_preference, apply_preferences

"""
    Preference

Structural preference with weight and constraint function.
"""
struct Preference
    name::Symbol
    weight::Float64
    constraint::Function
end

"""
    evaluate_preference(pref::Preference, state) -> Float64

Evaluate how well state satisfies preference.
"""
function evaluate_preference(pref::Preference, state)
    try
        score = pref.constraint(state)
        return pref.weight * score
    catch
        return 0.0
    end
end

"""
    apply_preferences(prefs::Vector{Preference}, state) -> Float64

Compute total preference score: Σᵢ wᵢ × constraint_i(s)
"""
function apply_preferences(prefs::Vector{Preference}, state)
    sum(evaluate_preference(p, state) for p in prefs)
end

# Default preferences
const DEFAULT_PREFERENCES = [
    Preference(:locality, 2.0, s -> -s.locality_score),
    Preference(:compactness, 1.0, s -> -length(s.placement) / 100.0),
]

end # module StructuralPreferences
