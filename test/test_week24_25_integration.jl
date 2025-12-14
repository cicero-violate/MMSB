using Test
using Base.Filesystem: cp
using ..MMSB

const IntentLowering = MMSB.IntentLowering
const UpsertPlan = MMSB.UpsertPlan
const DeltaTypes = MMSB.DeltaTypes
const API = MMSB.API
const TLog = MMSB.TLog
const ReplayEngine = MMSB.ReplayEngine

function _temp_state()
    config = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=false, tlog_path=tempname())
    MMSB.MMSBStateTypes.MMSBState(config)
end

function _make_plan(page, payload; predicate=data -> true, intent_id::AbstractString="intent")
    mask = Vector{Bool}(trues(length(payload)))
    metadata = Dict(:intent_id => intent_id)
    UpsertPlan.UpsertPlan(
        "select * from page $(Int(page.id))",
        predicate,
        UpsertPlan.DeltaSpec(UInt64(page.id), Vector{UInt8}(payload), mask),
        metadata,
    )
end

function _run_intent_benchmark(iterations::Int)
    state = _temp_state()
    page = API.create_page(state; size=16)
    payload = fill(UInt8(0xaa), 16)
    timings = Float64[]
    for i in 1:iterations
        intent_id = "bench-$(i)"
        plan = _make_plan(page, payload .+ UInt8(i); intent_id=intent_id)
        push!(timings, @elapsed IntentLowering.execute_upsert_plan!(state, plan))
    end
    entries = TLog.query_log(state)
    (iterations=iterations,
     total_time=sum(timings),
     avg_latency=sum(timings) / iterations,
     log_entries=length(entries))
end

@testset "Week 24 INT.G1 Integration" begin
    @testset "Intent → Upsert → Delta pipeline" begin
        state = _temp_state()
        page = API.create_page(state; size=4)
        plan = _make_plan(page, UInt8[0x01, 0x02, 0x03, 0x04];
                          predicate=data -> sum(data) == 0,
                          intent_id="week24-pipeline")
        result = IntentLowering.execute_upsert_plan!(state, plan)
        @test result.applied
        @test API.query_page(state, page.id) == plan.deltaspec.payload

        entries = TLog.query_log(state)
        @test length(entries) == 1
        metadata = DeltaTypes.intent_metadata(entries[1]; parse=true)
        @test metadata[:intent_id] == "week24-pipeline"
        @test metadata[:query] == plan.query
        @test metadata[:delta_id] == Int(entries[1].id)
    end

    @testset "QMU separation semantics" begin
        state = _temp_state()
        page = API.create_page(state; size=4)
        API.update_page(state, page.id, UInt8[0x00, 0x01, 0x02, 0x03])
        baseline = API.query_page(state, page.id)

        block_plan = _make_plan(page, UInt8[0x05, 0x06, 0x07, 0x08];
                                predicate=data -> sum(data) > 255,
                                intent_id="week24-qmu-block")
        blocked = IntentLowering.execute_upsert_plan!(state, block_plan)
        @test !blocked.applied
        @test blocked.query_snapshot == baseline
        @test API.query_page(state, page.id) == baseline

        allow_plan = _make_plan(page, UInt8[0x09, 0x0a, 0x0b, 0x0c];
                                predicate=data -> sum(data) >= 0,
                                intent_id="week24-qmu-allow")
        allowed = IntentLowering.execute_upsert_plan!(state, allow_plan)
        @test allowed.applied
        @test API.query_page(state, page.id) == allow_plan.deltaspec.payload
    end

    @testset "TLog persistence with metadata" begin
        state = _temp_state()
        page = API.create_page(state; size=2)
        for idx in 1:3
            plan = _make_plan(page, UInt8[idx, idx + 1]; intent_id="persist-$(idx)")
            IntentLowering.execute_upsert_plan!(state, plan)
        end
        entries = TLog.query_log(state)
        @test length(entries) == 3
        metas = [DeltaTypes.intent_metadata(delta; parse=true) for delta in entries]
        @test metas[1][:intent_id] == "persist-1"
        @test metas[end][:intent_id] == "persist-3"
        @test metas[end][:delta_id] == Int(entries[end].id)
    end

    @testset "Validation rejects malformed deltas" begin
        state = _temp_state()
        page = API.create_page(state; size=4)
        mask = UInt8[1, 0, 1, 0]
        payload = UInt8[0xaa, 0xbb]
        bad_delta = MMSB.DeltaTypes.Delta(UInt64(42), UInt64(page.id), UInt32(1), mask, payload, :test; is_sparse=false)
        @test_throws MMSB.RustErrors.RustFFIError MMSB.DeltaRouter.route_delta!(state, bad_delta)
        @test isempty(TLog.query_log(state))
    end

    @testset "Performance benchmark capture" begin
        bench = _run_intent_benchmark(5)
        @test bench.iterations == bench.log_entries
        @test bench.total_time > 0
        @test bench.avg_latency > 0
    end
end

@testset "Week 25 INT.G2 Replay Verification" begin
    state = _temp_state()
    page = API.create_page(state; size=4)

    plan_a = _make_plan(page, UInt8[0x10, 0x11, 0x12, 0x13]; intent_id="week25-A")
    plan_b = _make_plan(page, UInt8[0x20, 0x21, 0x22, 0x23]; intent_id="week25-B")
    plan_c = _make_plan(page, UInt8[0x30, 0x31, 0x32, 0x33]; intent_id="week25-C")

    IntentLowering.execute_upsert_plan!(state, plan_a)
    IntentLowering.execute_upsert_plan!(state, plan_b)
    IntentLowering.execute_upsert_plan!(state, plan_c)

    @testset "Basic replay with intent metadata" begin
        final_state = API.query_page(state, page.id)
        replayed = ReplayEngine.replay_to_epoch(state, typemax(UInt32))
        @test API.query_page(replayed, page.id) == final_state

        history = ReplayEngine.replay_with_predicate(state, (epoch, delta) -> begin
            meta = DeltaTypes.intent_metadata(delta; parse=true)
            meta !== nothing && haskey(meta, :intent_id)
        end)
        @test length(history) == 3
    end

    @testset "Intent filtering" begin
        filtered = ReplayEngine.replay_with_predicate(state, (epoch, delta) -> begin
            meta = DeltaTypes.intent_metadata(delta; parse=true)
            meta !== nothing && get(meta, :intent_id, "") == "week25-B"
        end)
        @test length(filtered) == 1
        meta = DeltaTypes.intent_metadata(filtered[1][2]; parse=true)
        @test meta[:intent_id] == "week25-B"
    end

    @testset "Intent causality linkage" begin
        entries = TLog.query_log(state)
        for delta in entries
            meta = DeltaTypes.intent_metadata(delta; parse=true)
            @test meta[:delta_id] == Int(delta.id)
            @test meta[:intent_id] != ""
        end
    end

    @testset "Checkpoint retains intent metadata" begin
        mktemp() do path, io
            close(io)
            TLog.checkpoint_log!(state, path)
            log_copy = tempname()
            cp(state.config.tlog_path, log_copy; force=true)
            state_after_cp = MMSB.MMSBStateTypes.MMSBState(MMSB.MMSBStateTypes.MMSBConfig(tlog_path=log_copy))
            TLog.load_checkpoint!(state_after_cp, path)
            original_deltas = TLog.query_log(state)
            restored_deltas = TLog.query_log(state_after_cp)
            @test !isempty(original_deltas)
            @test !isempty(restored_deltas)
            meta = DeltaTypes.intent_metadata(restored_deltas[end]; parse=true)
            @test meta[:intent_id] == "week25-C"
        end
    end

    @testset "Intent replay documentation scenario" begin
        history = ReplayEngine.replay_with_predicate(state, (epoch, delta) -> begin
            meta = DeltaTypes.intent_metadata(delta; parse=true)
            meta !== nothing && occursin("week25", meta[:intent_id])
        end)
        @test length(history) == 3
    end
end
