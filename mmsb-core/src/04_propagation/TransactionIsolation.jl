module TransactionIsolation
using ..PageTypes: PageID
using ..MMSBStateTypes: MMSBState
export Transaction, begin_transaction, commit_transaction, rollback_transaction
mutable struct Transaction
id::UInt64
epoch::UInt64
dirty_pages::Set{PageID}
end
const ACTIVE_TXN=IdDict{MMSBState,Transaction}()
begin_transaction(s)=(t=Transaction(rand(UInt64),0,Set{PageID}());ACTIVE_TXN[s]=t;t)
commit_transaction(s)=(haskey(ACTIVE_TXN,s)&&delete!(ACTIVE_TXN,s);true)
rollback_transaction(s)=(haskey(ACTIVE_TXN,s)&&delete!(ACTIVE_TXN,s);false)
with_transaction(f,s)=(t=begin_transaction(s);try;r=f(t);commit_transaction(s);r;catch e;rollback_transaction(s);rethrow(e);end)
end
