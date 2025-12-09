"""
World simulation using MMSB as state machine.
"""
module WorldSimulation

export World, Entity, simulate_step!, add_entity!

using ..MMSBStateTypes: MMSBState
using ..MMSBStateTypes: allocate_page_id!
using ..PageTypes: Page, PageID
using ..DeltaTypes: Delta

struct Entity
    id::PageID
    entity_type::Symbol
    properties::Dict{Symbol, Any}
end

mutable struct World
    mmsb_state::MMSBState
    entities::Dict{PageID, Entity}
    time::Float64
    dt::Float64
end

World(state::MMSBState, dt=0.01) = World(state, Dict{PageID, Entity}(), 0.0, dt)

function add_entity!(world::World, entity_type::Symbol, props::Dict{Symbol, Any})::Entity
    page_id = allocate_page_id!(world.mmsb_state)
    entity = Entity(page_id, entity_type, props)
    world.entities[page_id] = entity
    return entity
end

function simulate_step!(world::World)
    world.time += world.dt
    # Placeholder: Update entity physics/logic
    nothing
end

end # module
