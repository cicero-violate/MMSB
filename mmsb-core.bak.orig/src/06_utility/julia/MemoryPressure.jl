module MemoryPressure
using ..PageTypes: PageID
using ..MMSBStateTypes: MMSBState
export detect_memory_pressure, evict_lru_pages, record_access
mutable struct LRUTracker
access_times::Dict{PageID,UInt64}
access_counter::UInt64
end
LRUTracker()=LRUTracker(Dict{PageID,UInt64}(),0)
const LRU_TRACKERS=IdDict{MMSBState,LRUTracker}()
record_access(s,p)=(t=get!(LRU_TRACKERS,s,LRUTracker());t.access_counter+=1;t.access_times[p]=t.access_counter)
detect_memory_pressure(s,th=0.8)=false
function evict_lru_pages(s,n)
t=get!(LRU_TRACKERS,s,LRUTracker())
sorted=sort(collect(t.access_times),by=x->x[2])
evicted=PageID[]
for (p,_) in sorted[1:min(n,length(sorted))]
push!(evicted,p);delete!(t.access_times,p)
end
evicted
end
end
