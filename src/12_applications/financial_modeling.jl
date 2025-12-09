"""
Financial modeling using MMSB for state tracking.
"""
module FinancialModeling

export Portfolio, Asset, compute_value, rebalance!

using ..MMSBStateTypes: MMSBState
using ..PageTypes: PageID

struct Asset
    symbol::String
    quantity::Float64
    page_id::PageID
end

mutable struct Portfolio
    state::MMSBState
    assets::Dict{String, Asset}
    cash::Float64
end

Portfolio(state::MMSBState, cash=0.0) = Portfolio(state, Dict{String, Asset}(), cash)

function compute_value(portfolio::Portfolio, prices::Dict{String, Float64})::Float64
    value = portfolio.cash
    for (symbol, asset) in portfolio.assets
        value += asset.quantity * get(prices, symbol, 0.0)
    end
    return value
end

function rebalance!(portfolio::Portfolio, target_weights::Dict{String, Float64})
    # Placeholder: Compute trades to achieve target weights
    nothing
end

end # module
