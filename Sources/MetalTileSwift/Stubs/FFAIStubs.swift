import Metal

// Stub dispatch functions for kernels that FFAI references but the
// `tile emit` auto-discovery (via `inventory::iter` over
// `crates/metaltile-std/`) does not currently produce. They let the
// FFAI package compile; calling any of them at runtime traps.
//
// To remove a stub: (a) implement the kernel in
// `crates/metaltile-std/src/...` with an `inventory::submit!` block,
// (b) re-run `make regenerate-kernels`, (c) drop the matching stub
// func below.
//
// Previously-stubbed families now emitted by metaltile auto-discovery
// and therefore removed: `aura_dequant_rotated_*`, `aura_encode_*`
// (int{2,3,4,8} × {f32,f16,bf16}), and — as of metaltile #145 —
// `ffai_gemm_*`, `ffai_rope_yarn_*`, `ffai_sdpa_multi_*`,
// `mt_rms_norm_wide_*`. The only surviving stubs are the
// indirect-dispatch `dequant_gemv` variants below.

extension MetalTileKernels {
    @inline(never) private static func unimplemented(_ name: String) -> Never {
        fatalError("\(name): kernel not currently emitted. Add a Rust source under `crates/metaltile-std/` with an `inventory::submit!` block; `tile emit` picks them up via inventory::iter.")
    }

    // MARK: - dequant_gemv_int4_*_indirect (Day-1 GPU-router plumbing)
    //
    // Indirect-dispatch variants of dequant_gemv_int4 — same PSO + args
    // as the direct kernels but dispatch shape comes from an
    // `MTLBuffer` instead of `MTLSize`. The old `metaltile-emit/main.rs`
    // produced these via a custom Swift wrapper generator (commit
    // `b2eadca` on `feat/ffai-kernel-pack`); the new `tile emit`
    // auto-discovery path doesn't carry that custom logic.
    // `Ops.dequantGemvIndirect` references these but is **not** wired
    // into the production decode / prefill path today (the GPU router
    // is still host-side). These stubs unblock compile; restoring the
    // indirect path needs the wrapper logic ported into `tile emit`
    // or a hand-written wrapper here.

    public static func dequant_gemv_int4_f16_indirect(
        weight: MTLBuffer, weightOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        in_dim: UInt32, group_size: UInt32,
        indirectBuffer: MTLBuffer, indirectBufferOffset: Int,
        threadgroupSize: MTLSize, on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }

    public static func dequant_gemv_int4_bf16_indirect(
        weight: MTLBuffer, weightOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        input: MTLBuffer, inputOffset: Int = 0,
        output: MTLBuffer, outputOffset: Int = 0,
        in_dim: UInt32, group_size: UInt32,
        indirectBuffer: MTLBuffer, indirectBufferOffset: Int,
        threadgroupSize: MTLSize, on commandBuffer: MTLCommandBuffer
    ) { unimplemented(#function) }
}
