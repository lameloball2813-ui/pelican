//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    pub static_table_cbvs: Vec<super::StaticTableCbv>,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    pub heap_buffer_classes: u8,
    pub heap_demands: Vec<super::HeapDemand>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,

    /// The names of the emitted functions (which dx.ops ran); tests only.
    #[cfg(test)]
    pub op_names: Vec<String>,
    /// The emitted instructions, one line each; tests only.
    #[cfg(test)]
    pub insts: Vec<String>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    /// The logical-lane plan encountered an additional host wave operation.
    /// Re-run with the original hardware-wave-size requirement intact.
    RetryNativeLanes,
    Fail(ShaderError),
}

#[cfg(test)]
pub(crate) fn lds_allocation_audit(parts: &Parts) -> Option<(Site, usize, usize)> {
    let bad = HashMap::new();
    let l = Lower::new(parts, &bad);
    l.lds_allocation.filter(|&p| l.allocation_plan_proven(p))
        .map(|p| (p.init, p.gate, p.publish))
}

#[cfg(test)]
pub(crate) fn uniform_arithmetic_firstlane_audit(parts: &Parts) -> Vec<(usize, bool)> {
    let bad = HashMap::new();
    let l = Lower::new(parts, &bad);
    parts.p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32")
        .map(|i| (i.machine.offset, l.uniform_arithmetic_firstlane(i, &mut HashSet::new(), 0)))
        .collect()
}

#[cfg(test)]
pub(crate) fn partial_reciprocal_firstlane_audit(parts: &Parts) -> Vec<usize> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    l.find_sections();
    parts.p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32" && l.partial_reciprocal_firstlane(i))
        .map(|i| i.machine.offset).collect()
}

#[cfg(test)]
pub(crate) fn partial_data_conversion_firstlane_audit(parts: &Parts) -> Vec<usize> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    l.find_sections();
    parts.p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32" && l.partial_data_conversion_firstlane(i))
        .map(|i| i.machine.offset).collect()
}

#[cfg(test)]
pub(crate) fn uniform_scalar_select_firstlane_audit(parts: &Parts, plan_broadcasts: bool) -> Vec<usize> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    if plan_broadcasts {
        l.find_sections();
        l.plan_mesh_broadcasts();
    }
    parts.p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32" && l.uniform_scalar_select_at_firstlane(i))
        .map(|i| i.machine.offset).collect()
}

#[cfg(test)]
pub(crate) fn mesh_fetch_firstlane_audit(parts: &Parts) -> Vec<(usize, bool)> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    l.find_sections();
    l.plan_mesh_broadcasts();
    parts.p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32")
        .map(|i| (i.machine.offset, l.uniform_buffer_fetch_at_firstlane(i))).collect()
}

#[cfg(test)]
pub(crate) fn mesh_broadcast_audit(parts: &Parts) -> Vec<(usize, u32)> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    l.find_sections();
    l.plan_mesh_broadcasts();
    let mut sites: Vec<_> = l.mesh_broadcasts.iter().map(|(&(b, i), &slot)|
        (parts.p.blocks[b].insts[i].machine.offset, slot)).collect();
    sites.sort_unstable();
    sites
}

/// Native M0-relative waterfall loops that can be evaluated independently
/// in each lane. This certificate is not the resource-descriptor waterfall
/// certificate: here the selected scalar feeds M0 and a *relative register*
/// read, so the old EXEC retry must be checked in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RelativeReadWaterfall {
    relative: usize,
    index: Value,
}

fn relative_read_waterfalls(p: &Program, st: &Structured) -> HashMap<usize, RelativeReadWaterfall> {
    let mut certified = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        let [read, compare, save, m0, relative, subtract, retry] = block.insts.as_slice() else { continue };
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || block.preds.len() != 2 || !block.preds.contains(&b)
            || !matches!(retry.name(), "s_cbranch_execnz")
            || read.name() != "v_readfirstlane_b32"
            || compare.name() != "v_cmp_eq_u32"
            || save.name() != "s_and_saveexec_b64"
            || m0.name() != "s_mov_b32"
            || relative.name() != "v_movrels_b32"
            || subtract.name() != "s_xor_b64"
        {
            continue;
        }
        let Fields::Valu(read_f) = &read.fields else { continue };
        let [Operand::Vector { reg: index_reg, dwords: 1 }] = read_f.src.as_slice() else { continue };
        if read_f.neg != 0 || read_f.abs != 0 || read_f.opsel != 0 || read_f.neg_hi != 0
            || read_f.opsel_hi != 0 || read_f.clamp || read_f.omod != 0
            || read_f.sdwa.is_some() || read_f.dpp.is_some() { continue; }
        let Some(index) = read.read(Reg::V(*index_reg)) else { continue };
        let Some(scalar) = read.write(Reg::S(2)) else { continue };
        let Fields::Valu(f) = &compare.fields else { continue };
        if f.abs != 0 || f.neg != 0 || f.opsel != 0 || f.neg_hi != 0
            || f.opsel_hi != 0 || f.clamp || f.omod != 0
            || f.sdwa.is_some() || f.dpp.is_some()
            || !matches!(f.src.as_slice(), [Operand::Scalar { code: 2, dwords: 1 }, Operand::Vector { dwords: 1, .. }]
                | [Operand::Vector { dwords: 1, .. }, Operand::Scalar { code: 2, dwords: 1 }])
            || !compare.reads.iter().any(|&(r, v)| r == Reg::S(2) && v == scalar)
            || !compare.reads.iter().any(|&(r, v)| matches!(r, Reg::V(_)) && v == index)
        {
            continue;
        }
        let Fields::Valu(relative_f) = &relative.fields else { continue };
        if relative_f.neg != 0 || relative_f.abs != 0 || relative_f.opsel != 0
            || relative_f.neg_hi != 0 || relative_f.opsel_hi != 0
            || relative_f.clamp || relative_f.omod != 0
            || relative_f.sdwa.is_some() || relative_f.dpp.is_some()
            || !matches!(relative_f.src.as_slice(), [Operand::Vector { dwords: 1, .. }])
            || !matches!(relative_f.dst, Some(Operand::Vector { dwords: 1, .. }))
        { continue; }
        let (Some(cmp_lo), Some(cmp_hi)) = (compare.write(Reg::S(crate::ir::VCC)), compare.write(Reg::S(crate::ir::VCC+1))) else { continue };
        let (Some(saved_lo), Some(saved_hi)) = (save.write(Reg::S(crate::ir::VCC)), save.write(Reg::S(crate::ir::VCC+1))) else { continue };
        if save.read(Reg::S(crate::ir::VCC)) != Some(cmp_lo)
            || save.read(Reg::S(crate::ir::VCC+1)) != Some(cmp_hi)
            || m0.read(Reg::S(2)) != Some(scalar)
            || relative.read(Reg::S(crate::ir::M0)) != m0.write(Reg::S(crate::ir::M0))
            || !matches!(m0.fields, Fields::Sop { ref src, .. } if src.as_slice() == [Operand::Scalar { code: 2, dwords: 1 }])
            || relative.writes.iter().filter(|&&(r, _)| matches!(r, Reg::V(_))).count() != 1
        {
            continue;
        }
        let (Some(before_lo), Some(before_hi), Some(selected_lo), Some(selected_hi)) = (
            save.read(Reg::S(EXEC)), save.read(Reg::S(EXEC+1)),
            save.write(Reg::S(EXEC)), save.write(Reg::S(EXEC+1)))
        else { continue };
        if subtract.read(Reg::S(EXEC)) != Some(selected_lo)
            || subtract.read(Reg::S(EXEC+1)) != Some(selected_hi)
            || subtract.read(Reg::S(crate::ir::VCC)) != Some(saved_lo)
            || subtract.read(Reg::S(crate::ir::VCC+1)) != Some(saved_hi)
        {
            continue;
        }
        let same_src = |inst: &Inst, code: u8, index: usize| {
            matches!(&inst.fields, Fields::Sop { src, .. }
                if src.get(index) == Some(&Operand::Scalar { code, dwords: 2 }))
        };
        if !same_src(save, crate::ir::VCC, 0)
            || !same_src(subtract, EXEC, 0)
            || !same_src(subtract, crate::ir::VCC, 1)
            || retry.read(Reg::S(EXEC)) != subtract.write(Reg::S(EXEC))
            || retry.read(Reg::S(EXEC + 1)) != subtract.write(Reg::S(EXEC+1))
        { continue; }
        // The only value carried through this retry is the destination's
        // previously written VGPR. Replacing any other phi (e.g. a varying
        // scalar loop induction) with its preheader value is not valid.
        let Some(Operand::Vector { reg: dst, .. }) = relative_f.dst else { continue };
        if block.phis.len() != 3
            || block.phis.iter().filter(|phi| {
                matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC+1)
            }).count() != 2
        { continue; }
        let Some(phi) = block.phis.iter().find(|phi| phi.reg == Reg::V(dst)) else { continue };
        let Some(back) = block.preds.iter().position(|&pred| pred == b) else { continue };
        if phi.args.len() != 2 || phi.args[back] != relative.write(Reg::V(dst)).unwrap()
            || relative.read(Reg::V(dst)) != Some(phi.value)
        { continue; }
        let mask = |i: &Inst| st.mask_bits(i.read(Reg::S(EXEC))?, i.read(Reg::S(EXEC+1)));
        let Some(remaining) = mask(read) else { continue };
        let Some(selected) = mask(relative) else { continue };
        let (Some(retry_lo), Some(retry_hi)) =
            (subtract.write(Reg::S(EXEC)), subtract.write(Reg::S(EXEC+1))) else { continue };
        let Some(retry_mask) = st.mask_bits(retry_lo, Some(retry_hi)) else { continue };
        let Some(saved) = st.mask_bits(before_lo, Some(before_hi)) else { continue };
        if st.guards[b][4] != Some(selected)
            || st.bools.meet(&[(remaining, true), (saved, false)])
            || st.bools.meet(&[(remaining, true), (selected, true), (saved, false)])
            || st.bools.meet(&[(remaining, true), (selected, true), (retry_mask, true)])
            || st.bools.meet(&[(remaining, true), (selected, false), (retry_mask, false)])
            || st.bools.meet(&[(st.reach[b], true), (selected, true), (remaining, false)])
        {
            continue;
        }
        // The scalar selector is used only to partition the lanes for this
        // relative read. No later branch, descriptor, or output can observe it.
        let selected_values = [scalar, m0.write(Reg::S(crate::ir::M0)).unwrap()];
        let mut users = 0;
        for (at_b, bb) in p.blocks.iter().enumerate() {
            if bb.phis.iter().any(|phi| phi.args.iter().any(|v| selected_values.contains(v))) { users = usize::MAX; break; }
            for (at_i, inst) in bb.insts.iter().enumerate() {
                if inst.reads.iter().any(|&(_,v)| selected_values.contains(&v)) {
                    if at_b != b || !matches!(at_i, 1 | 3 | 4) { users = usize::MAX; break; }
                    users += 1;
                }
            }
        }
        if users != 3 { continue; }
        certified.insert(b, RelativeReadWaterfall { relative: 4, index });
    }
    certified
}

#[cfg(test)]
pub(crate) fn relative_read_waterfalls_audit(parts: &Parts) -> Vec<usize> {
    let mut out: Vec<_> = relative_read_waterfalls(parts.p, parts.st).keys()
        .map(|&b| parts.p.blocks[b].offset).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
pub(crate) fn relative_read_waterfalls_fixture_audit(p: &Program, st: &Structured) -> Vec<usize> {
    let mut out: Vec<_> = relative_read_waterfalls(p, st).keys()
        .map(|&b| p.blocks[b].offset).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
pub(crate) fn logical_lane_audit(parts: &Parts) -> Vec<(usize, u32, bool)> {
    let bad = HashMap::new();
    let mut l = Lower::new(parts, &bad);
    l.find_sections();
    l.plan_logical_lane_sites();
    let mut sites: Vec<_> = l.logical_lane_sites.iter().map(|(&(b, i), &(base, publish))|
        (parts.p.blocks[b].insts[i].machine.offset, base, publish)).collect();
    sites.sort_unstable();
    sites
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_with_static_cbvs(parts, false)
}

pub(crate) fn lower_with_static_cbvs(parts: &Parts, static_cbvs: bool) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    let mut logical_lanes = true;
    loop {
        match Lower::new(parts, &bad).run(logical_lanes, static_cbvs) {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::RetryNativeLanes) => {
                assert!(logical_lanes);
                logical_lanes = false;
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    /// Bytes each root view holds, by root parameter (see [`Self::root_view_size`]).
    root_sizes: HashMap<usize, V>,
    static_table_cbvs: Vec<super::StaticTableCbv>,
    static_cbv_sites: HashMap<Site, usize>,
    static_cbv_headers: HashMap<usize, (V, V)>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    atomic_descriptor_loads: HashSet<Site>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// A certified one-wave compute LDS write/read phase boundary.
    implicit_lds_barrier: Option<(Site, u32)>,
    /// Reusable LDS tiles with an all-group publish and retire per iteration.
    matrix_lds_barriers: Option<(Site, Site)>,
    /// Certified one-wave LDS allocator, including a group-wide cap abort.
    lds_allocation: Option<LdsAllocation>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Dedicated LDS dword base for each certified once-only mesh broadcast.
    mesh_broadcasts: HashMap<Site, u32>,
    /// Full, straight-line compute wave: private LDS snapshots of VGPR SSA
    /// sources. Each source is published exactly once; all its reads share it.
    logical_lane_sites: HashMap<Site, (u32, bool)>,
    relative_waterfalls: HashMap<usize, RelativeReadWaterfall>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// IR blocks whose `s_barrier` is control-dependent on a non-uniform
    /// (per-lane) branch: emitting `SYNC_THREAD_GROUP` (bit 0) there
    /// deadlocks 32-lane hosts when warps diverge, so only the memory fence
    /// bits are emitted.
    fence_only_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

#[cfg(test)]
thread_local! {
    static ALLOCATION_TEST_LIMIT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn with_allocation_test_limit<T>(limit: u32, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<u32>);
    impl Drop for Restore {
        fn drop(&mut self) { ALLOCATION_TEST_LIMIT.with(|c| c.set(self.0)); }
    }
    assert!(limit > 0);
    let _restore = Restore(ALLOCATION_TEST_LIMIT.with(|c| c.replace(Some(limit))));
    f()
}

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Certify at whole-table granularity. Removing a CBV from a fresh table
/// must not change ANY other possible use of its slot, including a dynamic
/// descriptor load, descriptor words used as data, or an escaped table base.
/// Static non-CBV entries in the same table retain their original indices.
fn static_cbv_plan(x: &Parts) -> (Vec<super::StaticTableCbv>, HashMap<Site, usize>) {
    use super::StaticTableCbv;
    let p = x.p;
    let res = x.res;
    let native = x.native;
    if native.layout.iter().any(|r| r.slot == super::STATIC_CBV_SCRATCH) {
        return Default::default();
    }
    let address: HashMap<_, _> = res.scalar.iter().copied().collect();
    let at = |(b, i): Site| &p.blocks[b].insts[i];
    let root_slot = |a: &Address| native.layout.get(a.param)
        .filter(|r| r.root == RootType::Table).map(|r| r.slot);
    let mut users = HashMap::<Value, Vec<Site>>::new();
    let mut phi_users = HashSet::new();
    for (b, bb) in p.blocks.iter().enumerate() {
        for phi in &bb.phis { phi_users.extend(phi.args.iter().copied()); }
        for (i, inst) in bb.insts.iter().enumerate() {
            for &(_, v) in &inst.reads { users.entry(v).or_default().push((b, i)); }
        }
    }
    let mut table_slots: BTreeMap<u8, Vec<&UsedResource>> = BTreeMap::new();
    for u in native.used.iter().filter(|u| u.root == RootType::Table) {
        table_slots.entry(u.slot).or_default().push(u);
    }
    let mut plans = BTreeMap::<StaticTableCbv, Vec<Site>>::new();
    'tables: for (slot, entries) in table_slots {
        if !entries.iter().any(|u| u.kind == ResourceType::Cbv) { continue; }
        let mut named = BTreeMap::new();
        for u in entries {
            if !matches!(u.kind, ResourceType::Cbv | ResourceType::Srv | ResourceType::Uav)
                || !matches!(u.dwords, 4 | 8) || u.table_offset_dwords % 8 != 0
                || (u.kind == ResourceType::Cbv && (u.dwords != 4 || u.stride != 0))
                || !native.layout.iter().any(|r| r.root == RootType::Table && r.slot == slot
                    && r.user_data == u.user_data && r.sgprs == 1)
                || named.insert(u.table_offset_dwords / 8, u).is_some() {
                continue 'tables;
            }
        }
        let mut loads = HashMap::<Site, StaticTableCbv>::new();
        let mut table_loads = HashSet::new();
        // A table address may be copied or computed, but every memory use
        // must be a tracked static scalar descriptor load in named slots.
        for &(site, a) in &res.scalar {
            if root_slot(&a) != Some(slot) { continue; }
            let inst = at(site);
            let words = match inst.name() {
                "s_load_dword" => 1,
                "s_load_dwordx2" => 2,
                "s_load_dwordx4" => 4,
                "s_load_dwordx8" => 8,
                "s_load_dwordx16" => 16,
                _ => continue 'tables,
            };
            if a.dynamic.is_some() || a.offset % 4 != 0 || res.waterfall.loads.contains(&site) {
                continue 'tables;
            }
            let Some(end) = a.offset.checked_add(words * 4 - 1) else { continue 'tables };
            for k in a.offset / 32..=end / 32 {
                let Some(u) = named.get(&k) else { continue 'tables };
                if u.kind == ResourceType::Cbv {
                    // No multi-descriptor fetch, partial descriptor load or
                    // range/array interpretation for a promoted CBV.
                    if words != 4 || a.offset % 32 != 0 || a.offset / 4 != u.table_offset_dwords {
                        continue 'tables;
                    }
                    loads.insert(site, StaticTableCbv { slot, table_offset_dwords: u.table_offset_dwords });
                }
            }
            table_loads.insert(site);
        }
        if loads.is_empty() { continue; }
        // Reject any path from a table's user SGPRs to an untracked load,
        // ordinary value, heap index, etc. Copies/constant address adds
        // propagate the taint; only the validated loads terminate it.
        let mut taint: HashSet<_> = res.params.iter().filter_map(|&(v, param, _)| {
            let r = native.layout.get(param)?;
            (r.root == RootType::Table && r.slot == slot).then_some(v)
        }).collect();
        if taint.is_empty() { continue; }
        loop {
            let old = taint.len();
            for (b, bb) in p.blocks.iter().enumerate() {
                for phi in &bb.phis {
                    if phi.args.iter().any(|v| taint.contains(v)) { taint.insert(phi.value); }
                }
                for (i, inst) in bb.insts.iter().enumerate() {
                    if !inst.reads.iter().any(|&(_, v)| taint.contains(&v)) { continue; }
                    if table_loads.contains(&(b,i)) { continue; }
                    if !matches!(inst.name(), "s_mov_b32" | "s_mov_b64" | "s_add_u32" | "s_add_i32" | "s_addc_u32") {
                        continue 'tables;
                    }
                    taint.extend(inst.writes.iter().map(|&(_,v)| v));
                }
            }
            if taint.len() == old { break; }
        }
        let mut sites = HashMap::<Site, Site>::new();
        let mut local = BTreeMap::<StaticTableCbv, Vec<Site>>::new();
        for d in &res.descriptors {
            for leaf in d.source.leaves() {
                let (a, first, loaded) = match *leaf {
                    Source::Loaded { load, first } => (address.get(&load), first, true),
                    Source::Fetched { load, first } => (address.get(&load), first, false),
                    Source::Built { ref address, .. } => (Some(address), 0, false),
                    _ => continue,
                };
                let Some(a) = a else { continue 'tables };
                if root_slot(a) != Some(slot) { continue; }
                if !loaded || a.dynamic.is_some() || d.role != Role::Resource { continue 'tables; }
                let Some(byte) = a.offset.checked_add(u32::from(first) * 4) else { continue 'tables };
                let Some(u) = named.get(&(byte / 32)) else { continue 'tables };
                if byte % 32 != 0 { continue 'tables; }
                if u.kind != ResourceType::Cbv { continue; }
                let Source::Loaded { load, first: 0 } = &d.source else { continue 'tables };
                let load = *load;
                let Some(&key) = loads.get(&load) else { continue 'tables };
                let inst = at(d.site);
                let (code, soffset) = match &inst.fields {
                    Fields::Smem { base: Operand::Scalar { code, dwords: 4 }, soffset, .. }
                        if matches!(inst.name(), "s_buffer_load_dword" | "s_buffer_load_dwordx2" | "s_buffer_load_dwordx4" | "s_buffer_load_dwordx8" | "s_buffer_load_dwordx16") => (*code, soffset),
                    Fields::Buffer { rsrc: Operand::Scalar { code, dwords: 4 }, soffset, format: None, lds: false, tfe: false, .. }
                        if matches!(inst.name(), "buffer_load_dword" | "buffer_load_dwordx2" | "buffer_load_dwordx3" | "buffer_load_dwordx4") => (*code, soffset),
                    _ => continue 'tables,
                };
                if code > 102 || matches!(*soffset, Operand::Scalar { code: r, dwords } if r < code+4 && code < r.saturating_add(dwords)) {
                    continue 'tables;
                }
                let desc = at(load);
                let Fields::Smem { data: Some(Operand::Scalar { code: dc @ 0..=102, dwords: 4 }), .. } = desc.fields else { continue 'tables };
                if (0..4).any(|k| desc.write(Reg::S(dc+k)).is_none() || desc.write(Reg::S(dc+k)) != inst.read(Reg::S(code+k))) {
                    continue 'tables;
                }
                if sites.insert(d.site, load).is_some() { continue 'tables; }
                local.entry(key).or_default().push(d.site);
            }
        }
        // Every use of every CBV descriptor dword must be the exact
        // descriptor operand of a certified read, not a phi or address.
        for &load in loads.keys() {
            for &(_, v) in &at(load).writes {
                if phi_users.contains(&v) || users.get(&v).is_some_and(|uses|
                    uses.iter().any(|site| sites.get(site) != Some(&load))) { continue 'tables; }
            }
        }
        // Promote all CBVs in this table together, or none. In particular,
        // leaving one unknown CBV would still force a fresh allocation.
        if named.values().filter(|u| u.kind == ResourceType::Cbv).any(|u|
            !local.contains_key(&StaticTableCbv { slot, table_offset_dwords: u.table_offset_dwords })) { continue; }
        plans.extend(local);
    }
    let mut keys = Vec::new();
    let mut sites = HashMap::new();
    // Bound the metadata and worst-case scratch size (<= 16 MiB).
    if plans.len() > 256 { return Default::default(); }
    for (key, uses) in plans {
        let index = keys.len();
        keys.push(key);
        for site in uses { sites.insert(site, index); }
    }
    (keys, sites)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LdsAllocation {
    init: Site,
    gate: usize,
    publish: usize,
    abort_slot: u32,
    reciprocal: Option<Site>,
}

/// The only effects before the convergence point are the lane-zero clear
/// and at most one atomic increment per participating lane. The original
/// allocator's count is at byte 4, its published base at byte 0.
fn certify_lds_allocation(x: &Parts) -> Option<LdsAllocation> {
    let StageIo::Cs(cs) = x.native.io else { return None };
    if !x.native.wave64 || !x.p.wave64 || cs.threads != [64, 1, 1]
        || cs.lds_bytes() != 512 || cs.thread_id_vgprs() != 1 { return None; }
    let p = x.p;
    let st = x.st;
    let mut ds = Vec::new();
    let mut atomic = None;
    let mut reciprocal = None;
    for (b, block) in p.blocks.iter().enumerate() {
        if !super::wave::sections(p, b).is_empty() { return None; }
        for (i, inst) in block.insts.iter().enumerate() {
            let n = inst.name();
            if matches!(&inst.fields, Fields::Ds { .. }) { ds.push((b, i)); }
            else if n == "buffer_atomic_add" {
                if atomic.replace((b, i)).is_some() { return None; }
            } else if n == "v_readfirstlane_b32" {
                if reciprocal.replace((b, i)).is_some() { return None; }
            } else if n.contains("atomic") || n == "s_barrier"
                || n.starts_with("s_sendmsg") || matches!(inst.fields, Fields::Exp { .. })
                || matches!(inst.fields, Fields::Flat { .. }) { return None; }
        }
    }
    let [init, increment, count, base, load] = ds.as_slice() else { return None };
    let (init, increment, count, base, load) = (*init, *increment, *count, *base, *load);
    let atomic = atomic?;
    let at = |(b, i): Site| &p.blocks[b].insts[i];
    let exec = |site: Site| {
        let i = at(site);
        st.mask_bits(i.read(Reg::S(EXEC))?, i.read(Reg::S(EXEC + 1)))
    };
    let plain = |i: &Inst| matches!(&i.fields, Fields::Valu(v)
        if v.neg == 0 && v.abs == 0 && v.opsel == 0 && v.neg_hi == 0 && v.opsel_hi == 0
        && !v.clamp && v.omod == 0 && v.sdwa.is_none() && v.dpp.is_none() && v.carry.is_none());
    let value = |i: &Inst, op: &Operand| match *op {
        Operand::Vector { reg, dwords: 1 } => i.read(Reg::V(reg)),
        _ => None,
    };
    let const_at = |site: Site, v: Value, k: u8| {
        let (b, i) = site;
        p.blocks[b].insts[..i].iter().enumerate().any(|(j, w)|
            w.name() == "v_mov_b32" && plain(w)
                && matches!(&w.fields, Fields::Valu(f) if f.src.as_slice() == [Operand::Const(k)])
                && w.writes.iter().any(|&(r, a)| matches!(r, Reg::V(_)) && a == v)
                && exec((b, j)) == exec(site) && st.guards[b][j] == st.guards[b][i])
    };
    if init.0 != 0 || !p.blocks[0].preds.is_empty() || st.reach[0] != Bool::TRUE { return None; }
    // Clear only from the genuine local thread zero.
    let lane0 = exec(init)?;
    if st.guards[0][init.1] != Some(lane0) { return None; }
    let cmp = p.blocks[0].insts[..init.1].iter().enumerate().find(|(j, w)| {
        if w.name() != "v_cmp_eq_u32" || !plain(w) || exec((0, *j)) != Some(Bool::TRUE)
            || st.guards[0][*j].is_some() { return false; }
        let Fields::Valu(v) = &w.fields else { return false };
        let [Operand::Const(128), Operand::Vector { reg, dwords: 1 }] = v.src.as_slice() else { return false };
        if !w.read(Reg::V(*reg)).is_some_and(|a| x.io.inputs.contains(&(a, Input::ThreadIdInGroup(0)))) { return false; }
        let Some(Operand::Scalar { code, dwords: 2 }) = v.dst else { return false };
        w.write(Reg::S(code)).is_some_and(|lo| st.mask_bits(lo, w.write(Reg::S(code+1))) == Some(lane0))
    })?;
    let Fields::Valu(cmp_v) = &cmp.1.fields else { return None };
    let Some(Operand::Scalar { code, .. }) = cmp_v.dst else { return None };
    let cmp_lo = cmp.1.write(Reg::S(code))?;
    if st.bools.node(lane0) != Some((Atom::Lane(cmp_lo), Bool::FALSE, Bool::TRUE)) { return None; }
    let Fields::Ds { addr: Some(a), data, offset0: 0, offset1: 1, gds: false, .. } = &at(init).fields else { return None };
    if at(init).name() != "ds_write2_b32" || data.len() != 2
        || !const_at(init, value(at(init), a)?, 128)
        || !data.iter().all(|d| value(at(init), d).is_some_and(|v| const_at(init, v, 128))) { return None; }
    let restore = p.blocks[0].insts.get(init.1+1)?;
    if restore.name() != "s_mov_b64" || st.mask_bits(restore.write(Reg::S(EXEC))?, restore.write(Reg::S(EXEC+1))) != Some(Bool::TRUE) { return None; }
    let init_barrier = (0, init.1+2);
    if init_barrier.1 >= p.blocks[0].insts.len() { return None; }
    let Fields::Ds { addr: Some(a), data, offset0: 4, offset1: 0, gds: false, .. } = &at(increment).fields else { return None };
    if at(increment).name() != "ds_add_rtn_u32" || data.len() != 1
        || !const_at(increment, value(at(increment), a)?, 128)
        || !const_at(increment, value(at(increment), &data[0])?, 129)
        || st.guards[increment.0][increment.1].is_some() { return None; }
    let leader = count.0;
    if base.0 != leader || atomic.0 != leader || !(count.1 < atomic.1 && atomic.1 < base.1)
        || [count, atomic, base].iter().any(|&(b,i)| st.guards[b][i].is_some() || exec((b,i)) != Some(lane0)) { return None; }
    let [gate] = p.blocks[leader].preds.as_slice() else { return None };
    let gate = *gate;
    let LaneExit::Jump(publish) = st.exits[leader] else { return None };
    let LaneExit::Branch { taken, not_taken, when } = st.exits[gate] else { return None };
    let leader_taken = taken == leader && not_taken == publish;
    if !leader_taken && !(not_taken == leader && taken == publish) { return None; }
    // The branch is precisely the lane-zero predicate on all reachable lanes.
    if st.bools.meet(&[(st.reach[gate], true), (when, leader_taken), (lane0, false)])
        || st.bools.meet(&[(st.reach[gate], true), (when, !leader_taken), (lane0, true)]) { return None; }
    for b in [gate, publish] {
        let w = p.blocks[b].insts.first()?;
        if w.name() != "s_mov_b64" || st.mask_bits(w.write(Reg::S(EXEC))?, w.write(Reg::S(EXEC+1))) != Some(Bool::TRUE) { return None; }
    }
    let Fields::Ds { addr: Some(a), dst: Some(Operand::Vector { reg: count_reg, dwords: 1 }), offset0: 4, offset1: 0, gds: false, .. } = &at(count).fields else { return None };
    if at(count).name() != "ds_read_b32" || !const_at(count, value(at(count), a)?, 128) { return None; }
    let Fields::Buffer { data: Some(Operand::Vector { reg: result_reg, dwords: 1 }), soffset: Operand::Const(128), offset: 0, idxen, offen, cache, lds: false, tfe: false, format: None, .. } = &at(atomic).fields else { return None };
    if idxen == offen || !cache.glc || at(count).write(Reg::V(*count_reg))? != at(atomic).read(Reg::V(*result_reg))? { return None; }
    let Fields::Ds { addr: Some(a), data, offset0: 0, offset1: 0, gds: false, .. } = &at(base).fields else { return None };
    if at(base).name() != "ds_write_b32" || data.len() != 1
        || !const_at(base, value(at(base), a)?, 128)
        || value(at(base), &data[0])? != at(atomic).write(Reg::V(*result_reg))? { return None; }
    let Fields::Ds { addr: Some(a), offset0: 0, offset1: 0, gds: false, .. } = &at(load).fields else { return None };
    if at(load).name() != "ds_read_b32" || !const_at(load, value(at(load), a)?, 128) { return None; }
    // Verify the actual lowered CFG against the graph used by loop counters
    // and SSA phis. The prefix may loop, but each back edge is capped; a
    // conditional cap or unconditional cap goes directly to the gate.
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    if order.len() != n || gate == 0 { return None; }
    let mut rpo = vec![usize::MAX; n];
    for (i,&b) in order.iter().enumerate() { rpo[b] = i; }
    let succ = |b: usize| -> Vec<usize> { match st.exits[b] {
        LaneExit::Jump(t) => vec![t],
        LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
        _ => vec![],
    }};
    for b in 0..n {
        let mut a = succ(b);
        let mut c = p.blocks[b].exit.successors();
        a.sort_unstable(); c.sort_unstable();
        if a != c || a.iter().any(|&t| t >= n || !p.blocks[t].preds.contains(&b)) { return None; }
    }
    let reachable = |start: usize, stop: usize| {
        let mut seen = HashSet::new();
        let mut todo = vec![start];
        while let Some(b) = todo.pop() {
            if b == stop || !seen.insert(b) { continue; }
            todo.extend(succ(b));
        }
        seen
    };
    let pre = reachable(0, gate);
    let post = reachable(gate, usize::MAX);
    if pre.len()+post.len() != n || !pre.is_disjoint(&post)
        || !pre.contains(&increment.0) || !post.contains(&load.0)
        || succ(increment.0).iter().any(|&b| reachable(b, gate).contains(&increment.0))
        || reachable(leader, publish).len() != 1
        || !reachable(publish, usize::MAX).contains(&load.0) { return None; }
    // In a finite capped traversal an infinite path would revisit a
    // minimum-RPO header infinitely, without any lower-RPO entry to reset
    // its counter. This contradicts its cap. No native terminal precedes
    // the gate; after it the CFG is strictly acyclic.
    for &b in &pre {
        let s = succ(b);
        if s.is_empty() || s.iter().filter(|&&t| rpo[t] <= rpo[b]).count() > 1 { return None; }
        if p.blocks[b].insts.iter().any(|i| i.name().contains("store") || i.name().contains("atomic")) { return None; }
    }
    for &b in &post {
        if matches!(st.exits[b], LaneExit::Indirect) || succ(b).iter().any(|&t| rpo[t] <= rpo[b]) { return None; }
        if b != leader && b != publish && b != gate && !reachable(publish, usize::MAX).contains(&b) { return None; }
    }
    // An extra edge to the gate must not invalidate any SSA dominance.
    // The only definitions from the prefix usable after the gate are
    // entry definitions or incoming values of its phis (dead on abort).
    let defs: HashMap<_,_> = p.blocks.iter().enumerate().flat_map(|(b,bb)|
        bb.insts.iter().flat_map(move |i| i.writes.iter().map(move |&(_,v)| (v,b)))
            .chain(bb.phis.iter().map(move |phi| (phi.value,b)))).collect();
    let valid = |v: Value| defs.get(&v).is_none_or(|b| *b == 0 || !pre.contains(b));
    for &b in &post {
        let bb = &p.blocks[b];
        if bb.insts.iter().flat_map(|i| &i.reads).any(|&(_,v)| !valid(v)) { return None; }
        if b != gate && bb.phis.iter().flat_map(|phi| &phi.args).any(|&v| !valid(v)) { return None; }
        let mut masks: Vec<_> = st.guards[b].iter().flatten().copied().collect();
        if let LaneExit::Branch { when, .. } = st.exits[b] { masks.push(when); }
        for m in masks {
            if st.bools.support(m).iter().any(|a| match *a {
                Atom::Lane(v) | Atom::Uniform(v) | Atom::Phi(v) => !valid(v),
                _ => true,
            }) { return None; }
        }
    }
    Some(LdsAllocation { init: init_barrier, gate, publish, abort_slot: cs.lds_bytes()/4, reciprocal })
}

/// Prove a small, acyclic, one-wave LDS stencil has a race-free write phase
/// followed by a read-only phase. In particular, a barrier alone is NOT
/// enough if two different instructions write the same LDS address: a
/// physical wave orders those instructions, but two host waves need not.
/// Enumerate the exact 64 logical thread coordinates for the address proof.
fn implicit_lds_phase(x: &Parts) -> Option<(Site, usize, usize)> {
    let StageIo::Cs(cs) = x.native.io else { return None };
    if !x.native.wave64 || !x.p.wave64 || cs.threads != [4, 4, 4]
        || cs.lds_bytes() != 1024 || cs.thread_id_vgprs() != 3 { return None; }
    let p = x.p;
    let st = x.st;
    // Deliberately limited to two forward diamonds: conditional halo writes,
    // reconvergence, conditional consumers, final exit. No loop or bypass.
    let [entry, halo, join, consume, end] = p.blocks.as_slice() else { return None };
    let [LaneExit::Branch { taken: a, not_taken: b, .. }, LaneExit::Jump(2),
        LaneExit::Branch { taken: c, not_taken: d, .. }, LaneExit::Jump(4), LaneExit::End] = st.exits.as_slice() else { return None };
    if !((*a == 1 && *b == 2) || (*a == 2 && *b == 1))
        || !((*c == 3 && *d == 4) || (*c == 4 && *d == 3))
        || !entry.preds.is_empty() || halo.preds != [0] || join.preds != [0, 1]
        || consume.preds != [2] || end.preds != [2, 3]
        || st.reach[0] != Bool::TRUE || st.reach[2] != Bool::TRUE
        || !once_only_rendezvous(&st.exits, 2) { return None; }
    // Insert immediately AFTER the restored full EXEC, before any subsequent
    // instruction. The group rendezvous is independent of the prior mask.
    let restore = join.insts.first()?;
    let lo = restore.write(Reg::S(EXEC))?;
    let hi = restore.write(Reg::S(EXEC + 1))?;
    if join.insts.len() < 2 || restore.name() != "s_mov_b64"
        || st.mask_bits(lo, Some(hi)) != Some(Bool::TRUE)
        || st.guards[2][0].is_some() { return None; }
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    let mut local_load = None;
    let mut local_store = None;
    for (b, block) in p.blocks.iter().enumerate() {
        if !super::wave::sections(p, b).is_empty() { return None; }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            // No extra synchronization, lane broadcasts, atomics or other
            // side effects that could depend on wave-wide instruction order.
            if name == "s_barrier" || name.contains("readfirstlane") || name.contains("readlane")
                || name.contains("writelane") || name.contains("atomic") || name.starts_with("s_sendmsg")
                || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some())
                || name.contains("store") && !(b == 3 && name == "image_store")
                || matches!(inst.fields, Fields::Flat { .. } | Fields::Exp { .. } | Fields::Buffer { .. }) { return None; }
            if matches!(inst.fields, Fields::Image { .. }) {
                match (b, name) {
                    (0 | 1, "image_load") => {},
                    (3, "image_load") if local_load.is_none() => local_load = Some(i),
                    (3, "image_store") if local_store.is_none() => local_store = Some(i),
                    _ => return None,
                }
            }
            if let Fields::Ds { addr: Some(Operand::Vector { reg, dwords: 1 }), offset0, offset1,
                gds: false, .. } = &inst.fields {
                if st.guards[b][i].is_some() { return None; }
                let addr = inst.read(Reg::V(*reg))?;
                match name {
                    "ds_write_b32" if b < 2 => writes.push((b, addr, u32::from(*offset0) | u32::from(*offset1) << 8)),
                    "ds_read2_b32" if b == 3 => {
                        reads.push((addr, 4 * u32::from(*offset0)));
                        reads.push((addr, 4 * u32::from(*offset1)));
                    }
                    _ => return None,
                }
            } else if matches!(inst.fields, Fields::Ds { .. }) {
                return None;
            }
        }
    }
    if writes.is_empty() || reads.is_empty() { return None; }
    let defs: HashMap<_, _> = p.blocks.iter().enumerate().flat_map(|(b, block)|
        block.insts.iter().enumerate().flat_map(move |(i, inst)|
            inst.writes.iter().map(move |&(_, v)| (v, (b, i))))).collect();
    // The only UAV write is after the barrier. Its preceding image read is
    // at the same per-thread 3D texel of the same descriptor; no inter-thread
    // image read/write ordering is lost when the wave is split. Prove the
    // coordinates are groupId*4 + localId + 1 in each dimension.
    let (load_i, store_i) = (local_load?, local_store?);
    if load_i >= store_i || st.guards[3][load_i].is_some() || st.guards[3][store_i].is_some() { return None; }
    let load = &consume.insts[load_i];
    let store = &consume.insts[store_i];
    let image_key = |inst: &Inst| -> Option<([Value; 3], [Value; 8])> {
        let Fields::Image { addr, rsrc: Operand::Scalar { code, dwords: 8 }, dim: 2,
            unorm: true, a16: false, d16: false, tfe: false, lwe: false, .. } = &inst.fields else { return None; };
        let [Operand::Vector { reg, dwords: 3 }] = addr.as_slice() else { return None; };
        let mut coords = [Value(0); 3];
        let mut desc = [Value(0); 8];
        for (i, v) in coords.iter_mut().enumerate() { *v = inst.read(Reg::V(reg.checked_add(i as u8)?))?; }
        for (i, v) in desc.iter_mut().enumerate() { *v = inst.read(Reg::S(code.checked_add(i as u8)?))?; }
        Some((coords, desc))
    };
    let key = image_key(load)?;
    if key != image_key(store)? { return None; }
    for (c, v) in key.0.into_iter().enumerate() {
        let &(b, i) = defs.get(&v)?;
        let inst = &p.blocks[b].insts[i];
        let Fields::Valu(op) = &inst.fields else { return None; };
        let [Operand::Vector { reg, dwords: 1 }, Operand::Scalar { code, dwords: 1 }, Operand::Const(129)] = op.src.as_slice() else { return None; };
        if b != 0 || st.guards[b][i].is_some() || inst.name() != "v_add3_u32"
            || op.neg != 0 || op.abs != 0 || op.opsel != 0 || op.neg_hi != 0 || op.opsel_hi != 0
            || op.clamp || op.omod != 0 || op.sdwa.is_some() || op.dpp.is_some() || op.carry.is_some()
            || !x.io.inputs.contains(&(inst.read(Reg::V(*reg))?, Input::ThreadIdInGroup(c as u8))) { return None; }
        let &(sb, si) = defs.get(&inst.read(Reg::S(*code))?)?;
        let shift = &p.blocks[sb].insts[si];
        let Fields::Sop { src, .. } = &shift.fields else { return None; };
        let [Operand::Scalar { code, dwords: 1 }, Operand::Const(130)] = src.as_slice() else { return None; };
        if sb != 0 || si >= i || shift.name() != "s_lshl_b32"
            || !x.io.inputs.contains(&(shift.read(Reg::S(*code))?, Input::GroupId(c as u8))) { return None; }
    }
    let mut initialized = HashSet::new();
    let mut required = HashSet::new();
    for z in 0..4 { for y in 0..4 { for x_id in 0..4 {
        let mut eval = LdsAddressEval { x, defs: &defs, xyz: [x_id, y, z], memo: HashMap::new(), filter: false };
        for &(b, addr, offset) in &writes {
            if !eval.mask(st.reach[b], 0)? { continue; }
            let byte = eval.value(addr, 0)?.wrapping_add(offset);
            // Even same-lane writes to the same location are declined: this
            // proof needs no intra-phase memory ordering of any kind.
            if byte % 4 != 0 || byte >= cs.lds_bytes() || !initialized.insert(byte) { return None; }
        }
        // Overapproximate consumers by ALL threads, even if some will skip
        // the data-dependent read phase. Each read must have a producer.
        for &(addr, offset) in &reads {
            let byte = eval.value(addr, 0)?.wrapping_add(offset);
            if byte % 4 != 0 || byte >= cs.lds_bytes() { return None; }
            required.insert(byte);
        }
    } } }
    if !required.is_subset(&initialized) { return None; }
    Some(((2, 1), initialized.len(), required.len()))
}

/// The separable 8x8 filter stages load a 20x8 tile (two eight-wide
/// planes and a four-wide halo), then read its center and six symmetric
/// pairs. The native wave provides the write/read ordering implicitly.
/// Certify the actual SSA addresses and control flow, not the shader hash.
/// The scalar induction is limited to a single, two-iteration self-loop;
/// all producers precede a once-only full-EXEC group rendezvous.
fn filter_lds_phase(x: &Parts) -> Option<(Site, usize, usize)> {
    let StageIo::Cs(cs) = x.native.io else { return None };
    if !x.native.wave64 || !x.p.wave64 || cs.threads != [8, 8, 1]
        || cs.lds_bytes() != 1536 || cs.thread_id_vgprs() != 2 { return None; }
    let p = x.p;
    let st = x.st;
    let defs: HashMap<_, _> = p.blocks.iter().enumerate().flat_map(|(b, bb)|
        bb.insts.iter().enumerate().flat_map(move |(i, inst)|
            inst.writes.iter().map(move |&(_, v)| (v, (b, i))))).collect();
    let at = |(b, i): Site| &p.blocks[b].insts[i];
    let plain = |v: &crate::gfx10::Valu| v.abs == 0 && v.neg == 0 && v.opsel == 0
        && v.neg_hi == 0 && v.opsel_hi == 0 && !v.clamp && v.omod == 0
        && v.sdwa.is_none() && v.dpp.is_none() && v.carry.is_none();
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    let mut output = None;
    for (b, bb) in p.blocks.iter().enumerate() {
        if !super::wave::sections(p, b).is_empty() || st.exits[b] == LaneExit::Indirect { return None; }
        for (i, inst) in bb.insts.iter().enumerate() {
            let n = inst.name();
            if n.contains("atomic") || n.contains("readlane") || n.contains("readfirstlane")
                || n.contains("writelane") || n.contains("permute") || n.contains("mbcnt")
                || n.contains("swizzle") || n == "s_barrier" || n.starts_with("s_sendmsg")
                || reads_mask_as_data(st, inst)
                || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some()) { return None; }
            match &inst.fields {
                Fields::Ds { addr: Some(Operand::Vector { reg, dwords: 1 }), offset0, offset1, gds: false, .. } => {
                    if st.guards[b][i].is_some() { return None; }
                    let a = inst.read(Reg::V(*reg))?;
                    let offsets = [4 * u32::from(*offset0), 4 * u32::from(*offset1)];
                    match n {
                        "ds_write2_b32" => writes.push((b, i, a, offsets)),
                        "ds_read2_b32" => reads.push((b, i, a, offsets)),
                        _ => return None,
                    }
                }
                Fields::Ds { .. } | Fields::Buffer { .. } | Fields::Exp { .. } => return None,
                Fields::Image { .. } if n == "image_store" => {
                    if output.replace((b, i)).is_some() { return None; }
                }
                Fields::Image { .. } if n == "image_load" => {},
                Fields::Image { .. } => return None,
                Fields::Flat { .. } if n.starts_with("global_load_") => {
                    if !x.res.scalar.iter().any(|&(s, a)| s == (b, i) && a.param == CODE) { return None; }
                }
                Fields::Flat { .. } => return None,
                _ if n.contains("store") => return None,
                _ => {},
            }
        }
    }
    let [(base, _, _, _), (base2, _, _, _), (halo, _, _, _)] = writes.as_slice() else { return None };
    let (base, halo) = (*base, *halo);
    if *base2 != base || halo == base || !once_only_rendezvous(&st.exits, base) { return None; }
    let LaneExit::Jump(join) = st.exits[halo] else { return None };
    let LaneExit::Branch { when, taken, not_taken } = st.exits[base] else { return None };
    let halo_taken = taken == halo && not_taken == join;
    if !halo_taken && !(not_taken == halo && taken == join) { return None; }
    if p.blocks[halo].preds.as_slice() != [base] || p.blocks[join].preds.len() != 2
        || !p.blocks[join].preds.contains(&base) || !p.blocks[join].preds.contains(&halo)
        || !once_only_rendezvous(&st.exits, join) || st.reach[join] != Bool::TRUE { return None; }
    let restore = p.blocks[join].insts.first()?;
    if restore.name() != "s_mov_b64" || p.blocks[join].insts.len() < 2
        || st.guards[join][0].is_some() || st.mask_bits(restore.write(Reg::S(EXEC))?, restore.write(Reg::S(EXEC+1))) != Some(Bool::TRUE) { return None; }
    let mut after = HashSet::new();
    let mut todo = vec![join];
    while let Some(b) = todo.pop() {
        if !after.insert(b) { continue; }
        todo.extend(p.blocks[b].exit.successors());
    }
    if after.contains(&base) || after.contains(&halo) || reads.is_empty()
        || reads.iter().any(|&(b, _, _, _)| !after.contains(&b)) { return None; }
    let (ob, oi) = output?;
    if !after.contains(&ob) || st.guards[ob][oi].is_some() { return None; }
    // All other image/buffer data reads precede the rendezvous. Code-table
    // reads cannot alias a UAV. Output coordinates are one unique texel per
    // local thread and group, even if input and output views alias.
    for &b in &after {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            if inst.name() == "image_load" || inst.name().starts_with("s_buffer_load_") { return None; }
            // Scalar loads after the rendezvous may only read the immutable
            // code/table descriptor words, not data through a root view.
            if matches!(inst.fields, Fields::Smem { .. }) && !x.res.scalar.iter().any(|&(s, a)| {
                s == (b, i) && (a.param == CODE || x.native.layout.get(a.param).is_some_and(|r| r.root == RootType::Table))
            }) { return None; }
        }
    }
    let out = at((ob, oi));
    let Fields::Image { addr, dim: 1, a16: false, unorm: true, tfe: false, lwe: false, .. } = &out.fields else { return None };
    let coords: Vec<_> = addr.iter().flat_map(|a| match *a {
        Operand::Vector { reg, dwords } => (0..dwords).map(move |k| reg.checked_add(k)).collect::<Vec<_>>(),
        _ => vec![None],
    }).collect::<Option<_>>()?;
    if coords.len() != 2 { return None; }
    for (c, reg) in coords.into_iter().enumerate() {
        let &(b, i) = defs.get(&out.read(Reg::V(reg))?)?;
        let inst = at((b, i));
        let Fields::Valu(v) = &inst.fields else { return None; };
        let [Operand::Scalar { code, dwords: 1 }, Operand::Const(131), Operand::Vector { reg, dwords: 1 }] = v.src.as_slice() else { return None; };
        if after.contains(&b) || !once_only_rendezvous(&st.exits, b) || st.guards[b][i].is_some()
            || inst.name() != "v_lshl_add_u32" || !plain(v)
            || !x.io.inputs.contains(&(inst.read(Reg::S(*code))?, Input::GroupId(c as u8)))
            || !x.io.inputs.contains(&(inst.read(Reg::V(*reg))?, Input::ThreadIdInGroup(c as u8))) { return None; }
    }
    // A descriptor loaded from a static table, never a lane-dependent UAV.
    let desc = x.res.descriptors.iter().find(|d| d.site == (ob, oi) && d.role == Role::Resource)?;
    let Source::Loaded { load, first: 0 } = &desc.source else { return None; };
    if !x.res.scalar.iter().any(|&(s, a)| s == *load && a.dynamic.is_none()
        && x.native.layout.get(a.param).is_some_and(|r| r.root == RootType::Table)) { return None; }

    // One natural self-loop; no other backwards edge, including a revisit
    // of a write or output. The scalar phi starts at 1 and increments by 3
    // until the exact SCC comparison with 7 exits (values 1, 4).
    let back: Vec<_> = after.iter().flat_map(|&b| p.blocks[b].exit.successors().into_iter()
        .filter(move |&t| t <= b).map(move |t| (b, t))).collect();
    let [(loop_b, target)] = back.as_slice() else { return None };
    let loop_b = *loop_b;
    if *target != loop_b || loop_b == ob || p.blocks[loop_b].preds.len() != 2 { return None; }
    let bb = &p.blocks[loop_b];
    let LaneExit::Branch { when: loop_when, taken: lt, not_taken: lf } = st.exits[loop_b] else { return None; };
    let (Atom::Uniform(scc), low, high) = st.bools.node(loop_when)? else { return None; };
    if !((lt == loop_b && low == Bool::TRUE && high == Bool::FALSE)
        || (lf == loop_b && low == Bool::FALSE && high == Bool::TRUE)) { return None; }
    let &(cb, ci) = defs.get(&scc)?;
    let cmp = at((cb, ci));
    let Fields::Sop { src: cmp_src, .. } = &cmp.fields else { return None; };
    let [Operand::Scalar { code: sc, dwords: 1 }, Operand::Const(135)] = cmp_src.as_slice() else { return None; };
    if cb != loop_b || cmp.name() != "s_cmp_eq_u32" || *sc > 105 { return None; }
    let inc_value = cmp.read(Reg::S(*sc))?;
    let &(ib, ii) = defs.get(&inc_value)?;
    let inc = at((ib, ii));
    let Fields::Sop { src: inc_src, dst: Some(Operand::Scalar { code: dc, dwords: 1 }), .. } = &inc.fields else { return None; };
    let [Operand::Scalar { code: ic, dwords: 1 }, Operand::Const(131)] = inc_src.as_slice() else { return None; };
    if ib != loop_b || ii >= ci || inc.name() != "s_add_i32" || dc != sc || ic != sc { return None; }
    let induction = inc.read(Reg::S(*ic))?;
    let phi = bb.phis.iter().find(|phi| phi.value == induction && phi.reg == Reg::S(*ic))?;
    let back_i = bb.preds.iter().position(|&b| b == loop_b)?;
    if phi.args.len() != 2 || phi.args[back_i] != inc_value { return None; }
    let &(init_b, init_i) = defs.get(&phi.args[1-back_i])?;
    let init = at((init_b, init_i));
    if init_b != bb.preds[1-back_i] || init.name() != "s_mov_b32"
        || !matches!(&init.fields, Fields::Sop { src, .. } if src.as_slice() == [Operand::Const(129)]) { return None; }
    if reads.iter().any(|&(b, _, _, _)| b != loop_b && b != init_b) { return None; }

    let mut initialized = HashSet::new();
    let mut required = HashSet::new();
    for y in 0..8 { for x_id in 0..8 {
        let mut eval = LdsAddressEval { x, defs: &defs, xyz: [x_id, y, 0], memo: HashMap::new(), filter: true };
        for &(b, _, addr, offsets) in &writes {
            if b == halo && eval.mask(when, 0)? != halo_taken { continue; }
            for offset in offsets {
                let byte = eval.value(addr, 0)?.checked_add(offset)?;
                if byte % 4 != 0 || byte >= cs.lds_bytes() || !initialized.insert(byte) { return None; }
            }
        }
        for iv in [1, 4] {
            let mut eval = LdsAddressEval { x, defs: &defs, xyz: [x_id, y, 0], memo: HashMap::from([(induction, iv)]), filter: true };
            for &(_, _, addr, offsets) in &reads {
                for offset in offsets {
                    let byte = eval.value(addr, 0)?.checked_add(offset)?;
                    if byte % 4 != 0 || byte >= cs.lds_bytes() { return None; }
                    required.insert(byte);
                }
            }
        }
    } }
    if initialized.len() != 320 || required != initialized { return None; }
    Some(((join, 1), initialized.len(), required.len()))
}

/// The one-wave half-precision tiled matrix kernels load two 32x8 tiles
/// per outer iteration, then consume them in a four-iteration scalar loop.
/// Synchronize only after every cell has exactly one writer, and after all
/// consumers retire, before the next iteration can overwrite any cell.
/// This certificate is about CFG, uniform induction and LDS addresses, not
/// the arithmetic or identity of a particular game shader.
fn matrix_lds_phase(x: &Parts) -> Option<(Site, Site, usize, usize)> {
    let StageIo::Cs(cs) = x.native.io else { return None };
    if !x.native.wave64 || !x.p.wave64 || cs.threads != [8,8,1]
        || cs.lds_bytes() != 4096 || cs.thread_id_vgprs() != 2 { return None; }
    let p = x.p;
    let st = x.st;
    let n = p.blocks.len();
    let at = |(b,i): Site| &p.blocks[b].insts[i];
    let succ = |b: usize| -> Vec<usize> { match st.exits[b] {
        LaneExit::Jump(t) => vec![t],
        LaneExit::Branch {taken,not_taken,..} => vec![taken,not_taken],
        _ => vec![],
    }};
    let order = crate::ir::reverse_postorder(&p.blocks);
    if order.len() != n { return None; }
    let mut rank = vec![usize::MAX;n];
    for (i,&b) in order.iter().enumerate() { rank[b]=i; }
    for b in 0..n {
        let mut a = succ(b); let mut c = p.blocks[b].exit.successors();
        a.sort_unstable(); c.sort_unstable();
        if a != c || matches!(st.exits[b],LaneExit::Indirect) { return None; }
    }
    let mut writes = Vec::new(); let mut reads = Vec::new();
    let mut stores = Vec::new(); let mut fetches = Vec::new();
    for (b,bb) in p.blocks.iter().enumerate() {
        if !super::wave::sections(p,b).is_empty() { return None; }
        for (i,w) in bb.insts.iter().enumerate() {
            let name=w.name();
            if reads_mask_as_data(st,w) || name.contains("atomic") || name.contains("lane")
                || name.contains("permute") || name.starts_with("v_mbcnt") || name=="s_barrier"
                || name.starts_with("s_sendmsg") || matches!(&w.fields,Fields::Valu(v) if v.dpp.is_some())
                || matches!(w.fields,Fields::Flat{..}|Fields::Image{..}|Fields::Exp{..}) { return None; }
            match &w.fields {
                Fields::Ds {addr:Some(Operand::Vector{reg,dwords:1}),offset0,offset1,gds:false,..} => {
                    if st.guards[b][i].is_some() || st.mask_bits(w.read(Reg::S(EXEC))?,w.read(Reg::S(EXEC+1))) != Some(Bool::TRUE) {return None;}
                    let item=((b,i),w.read(Reg::V(*reg))?,[u32::from(*offset0)*4,u32::from(*offset1)*4]);
                    match name { "ds_write2_b32" => writes.push(item), "ds_read2_b32" => reads.push(item), _=>return None }
                },
                Fields::Ds{..} => return None,
                Fields::Buffer{lds:false,tfe:false,..} if name.starts_with("buffer_load_") || name.starts_with("tbuffer_load_") => fetches.push((b,i)),
                Fields::Buffer{lds:false,tfe:false,..} if name.starts_with("buffer_store_") => stores.push((b,i)),
                Fields::Buffer{..} => return None,
                _ if name.contains("store") => return None,
                _=>{},
            }
        }
    }
    if writes.len()!=8 || reads.len()!=16 || stores.is_empty() || fetches.is_empty() {return None;}
    let inner=reads[0].0.0;
    if reads.iter().any(|&(s,_,_)|s.0!=inner) {return None;}
    let back:Vec<_>=order.iter().flat_map(|&b| {
        let rank=&rank;
        succ(b).into_iter().filter(move|&t|rank[t]<=rank[b]).map(move|t|(b,t))
    }).collect();
    if back.len()!=2 || !back.contains(&(inner,inner)) {return None;}
    let (latch,outer)=*back.iter().find(|&&(b,t)| (b,t)!=(inner,inner))?;
    if outer==0 || outer==inner || latch==inner || rank[outer]>=rank[inner] || rank[inner]>=rank[latch] {return None;}
    let other = |b:usize,t:usize| -> Option<usize> { let s=succ(b); if s.len()!=2 || !s.contains(&t){None}else{s.into_iter().find(|&v|v!=t)} };
    if other(inner,inner)?!=latch || p.blocks[latch].preds!=[inner] {return None;}
    let done=other(latch,outer)?;
    if p.blocks[outer].preds.len()!=2 || !p.blocks[outer].preds.contains(&latch) {return None;}
    let pre=*p.blocks[outer].preds.iter().find(|&&b|b!=latch)?;
    if pre==0 || p.blocks[pre].preds!=[0] || succ(pre)!=[outer] {return None;}
    let bypass=other(0,pre)?;
    if p.blocks[inner].preds.len()!=2 || !p.blocks[inner].preds.contains(&inner) {return None;}
    let publish=*p.blocks[inner].preds.iter().find(|&&b|b!=inner)?;
    if succ(publish)!=[inner] {return None;}
    // All paths through a tile load reach its publication once. No side
    // entrance or exit, and every producer lies on every such path.
    let region = |start:usize,stop:usize| -> Option<HashSet<usize>> {
        let mut seen=HashSet::new(); let mut todo=vec![start];
        while let Some(b)=todo.pop() {
            if b==stop || !seen.insert(b) {continue;}
            let next=succ(b);
            if next.is_empty() || next.iter().any(|&t|rank[t]<=rank[b]) {return None;}
            todo.extend(next);
        }
        Some(seen)
    };
    let body=region(outer,inner)?;
    if !body.contains(&publish) || body.contains(&latch) || body.contains(&done) || body.contains(&bypass) {return None;}
    for &b in &body {
        if b!=outer && p.blocks[b].preds.iter().any(|q|!body.contains(q)) {return None;}
    }
    // A path reaching a producer's successor without the producer would
    // leave uninitialized cells; the stop-at-writer traversal must not
    // reach inner, and must have no terminal other than the writer.
    for &(s,_,_) in &writes {
        if !body.contains(&s.0) || region(outer,s.0).is_none_or(|seen|seen.contains(&inner)) {return None;}
    }
    let last=writes.iter().filter(|&&(s,_,_)|s.0==publish).map(|&(s,_,_)|s.1).max()?;
    // IR omits waitcnt/nop; publication can occur at the very end of the
    // producer block. The read loop's predecessor is the sole entry.
    let publish_site=(publish,last+1);
    let retire_site=(latch,0);
    if p.blocks[latch].insts.is_empty() {return None;}
    if fetches.iter().any(|&(b,_)| !body.contains(&b)) || stores.iter().any(|&(b,_)|body.contains(&b)||[0,pre,inner,latch].contains(&b)) {return None;}
    // Verify the only remaining regions are acyclic output/bypass code.
    let mut post=HashSet::new(); let mut todo=vec![done,bypass];
    while let Some(b)=todo.pop() {
        if !post.insert(b){continue;}
        if body.contains(&b)||[0,pre,inner,latch].contains(&b) {return None;}
        if succ(b).iter().any(|&t|rank[t]<=rank[b]) {return None;}
        todo.extend(succ(b));
    }
    if post.len()+body.len()+4!=n || stores.iter().any(|&(b,_)|!post.contains(&b)) {return None;}
    let defs:HashMap<_,_>=p.blocks.iter().enumerate().flat_map(|(b,bb)|bb.insts.iter().enumerate().flat_map(move|(i,w)|w.writes.iter().map(move|&(_,v)|(v,(b,i))))).collect();
    // Both loop exits are scalar comparisons whose operands have exact
    // uniform provenance. Do not treat every Atom::Uniform as sufficient:
    // a scalar value produced in a divergent branch may differ per lane.
    let cmp = |b:usize, on_true:usize, on_false:usize, name:&str| -> Option<&Inst> {
        let LaneExit::Branch{when,taken,not_taken}=st.exits[b] else{return None;};
        let (Atom::Uniform(scc),lo,hi)=st.bools.node(when)? else{return None;};
        if !((taken==on_true&&not_taken==on_false&&lo==Bool::FALSE&&hi==Bool::TRUE)
            ||(taken==on_false&&not_taken==on_true&&lo==Bool::TRUE&&hi==Bool::FALSE)){return None;}
        let &(cb,ci)=defs.get(&scc)?; let w=at((cb,ci));
        (cb==b && w.name()==name).then_some(w)
    };
    let scalar = |w:&Inst,k:usize| -> Option<Value> {
        let Fields::Sop{src,..}=&w.fields else{return None;};
        let Operand::Scalar{code:code@0..=105,dwords:1}=*src.get(k)? else{return None;};
        w.read(Reg::S(code))
    };
    let constant = |w:&Inst,k:usize| -> Option<u32> {
        let Fields::Sop{src,..}=&w.fields else{return None;};
        match *src.get(k)? {Operand::Const(c)=>crate::resource::inline(c),Operand::Literal(v)=>Some(v),_=>None}
    };
    let c=cmp(inner,latch,inner,"s_cmp_eq_u32")?;
    if constant(c,1)!=Some(8){return None;}
    let inc_val=scalar(c,0)?; let &(ib,ii)=defs.get(&inc_val)?; let inc=at((ib,ii));
    if ib!=inner || inc.name()!="s_add_i32" || constant(inc,1)!=Some(2){return None;}
    let induction=scalar(inc,0)?;
    let phi=p.blocks[inner].phis.iter().find(|ph|ph.value==induction && matches!(ph.reg,Reg::S(0..=105)))?;
    let self_i=p.blocks[inner].preds.iter().position(|&b|b==inner)?;
    if phi.args.len()!=2 || phi.args[self_i]!=inc_val{return None;}
    let &(init_b,init_i)=defs.get(&phi.args[1-self_i])?; let init=at((init_b,init_i));
    if init_b!=publish || init.name()!="s_mov_b32" || constant(init,0)!=Some(0){return None;}
    let first=cmp(0,bypass,pre,"s_cmp_ge_u32")?;
    let last_cmp=cmp(latch,done,outer,"s_cmp_ge_u32")?;
    let start=scalar(first,0)?; let bound=scalar(first,1)?;
    if scalar(last_cmp,1)?!=bound {return None;}
    let next=scalar(last_cmp,0)?; let &(nb,ni)=defs.get(&next)?; let add=at((nb,ni));
    if nb!=latch || add.name()!="s_add_i32" || constant(add,1)!=Some(32){return None;}
    let outer_iv=scalar(add,0)?;
    let phi=p.blocks[outer].phis.iter().find(|ph|ph.value==outer_iv && matches!(ph.reg,Reg::S(0..=105)))?;
    let li=p.blocks[outer].preds.iter().position(|&b|b==latch)?;
    if phi.args.len()!=2 || phi.args[li]!=next || phi.args[1-li]!=start{return None;}
    let &(sb,si)=defs.get(&start)?; let sh=at((sb,si));
    let &(bb,bi)=defs.get(&bound)?; let min=at((bb,bi));
    if sb!=0 || bb!=0 || sh.name()!="s_lshl_b32" || constant(sh,1)!=Some(8) || min.name()!="s_min_u32" {return None;}
    let end=scalar(min,0)?; let length=scalar(min,1)?;
    let &(eb,ei)=defs.get(&end)?;let end_add=at((eb,ei));
    if eb!=0 || end_add.name()!="s_add_i32" || scalar(end_add,0)?!=start || constant(end_add,1)!=Some(256){return None;}
    // Both bound inputs are from a static entry CBV, not a divergent UAV
    // load or fetched descriptor. Scalar inputs of this load are fixed.
    let batch=scalar(sh,0)?;
    let &(lb,li)=defs.get(&length)?; let &(ab,ai)=defs.get(&batch)?;
    if lb!=0 || ab!=0 || li!=ai {return None;}
    let load=at((lb,li));
    if load.name()!="s_buffer_load_dwordx4" || !matches!(&load.fields,Fields::Smem{offset:16,soffset:Operand::Scalar{code:NULL,..},..}) {return None;}
    let desc=x.res.descriptors.iter().find(|d|d.site==(lb,li)&&d.role==Role::Resource)?;
    let Source::Loaded{load:dl,first:0}=&desc.source else{return None;};
    let (_,a)=x.res.scalar.iter().find(|(s,a)|s==dl&&a.dynamic.is_none())?;
    let root=x.native.layout.get(a.param)?;
    if root.root!=RootType::Table || a.offset%32!=0 || !x.native.used.iter().any(|u|u.root==RootType::Table&&u.slot==root.slot&&u.kind==ResourceType::Cbv&&u.table_offset_dwords==a.offset/4){return None;}
    // Only the two induction values, which are equal in every group lane,
    // may control the barriers' iteration counts. Full EXEC at the write
    // and read sites ensures no missing writer or sleeping consumer.
    let mut initialized=HashSet::new(); let mut required=HashSet::new();
    for y in 0..8 {for tid in 0..8 {
        let mut ev=LdsAddressEval{x,defs:&defs,xyz:[tid,y,0],memo:HashMap::new(),filter:true};
        for &(_,addr,offsets) in &writes {for offset in offsets {
            let byte=ev.value(addr,0)?.checked_add(offset)?;
            if byte%4!=0||byte>=cs.lds_bytes()||!initialized.insert(byte){return None;}
        }}
        for iv in [0,2,4,6] {
            let mut ev=LdsAddressEval{x,defs:&defs,xyz:[tid,y,0],memo:HashMap::from([(induction,iv)]),filter:true};
            for &(_,addr,offsets) in &reads {for offset in offsets {
                let byte=ev.value(addr,0)?.checked_add(offset)?;
                if byte%4!=0||byte>=cs.lds_bytes(){return None;} required.insert(byte);
            }}
        }
    }}
    if initialized.len()!=1024||required!=initialized{return None;}
    Some((publish_site,retire_site,initialized.len(),required.len()))
}

/// Bounded evaluator for the address-only subset, not a shader interpreter.
/// Unknown instructions, data, masks, phis or modifiers decline certification.
struct LdsAddressEval<'a, 'b> {
    x: &'a Parts<'a>,
    defs: &'b HashMap<Value, Site>,
    xyz: [u32; 3],
    memo: HashMap<Value, u32>,
    // The filter certifier has proved a bounded scalar induction. Its
    // extra integer/float address operations are not admitted by the older
    // 3D implicit-LDS proof.
    filter: bool,
}

impl LdsAddressEval<'_, '_> {
    fn operand(&mut self, inst: &Inst, op: &Operand, depth: usize) -> Option<u32> {
        match *op {
            Operand::Const(c @ 128..=208) => crate::resource::inline(c),
            Operand::Literal(v) => Some(v),
            Operand::Vector { reg, dwords: 1 } => self.value(inst.read(Reg::V(reg))?, depth + 1),
            Operand::Scalar { code: code @ 0..=105, dwords: 1 } if self.filter =>
                self.value(inst.read(Reg::S(code))?, depth + 1),
            _ => None,
        }
    }

    fn mask(&mut self, mask: Bool, depth: usize) -> Option<bool> {
        if depth >= 96 { return None; }
        if mask == Bool::TRUE { return Some(true); }
        if mask == Bool::FALSE { return Some(false); }
        let (atom, lo, hi) = self.x.st.bools.node(mask)?;
        let Atom::Lane(v) = atom else { return None; };
        let set = self.value(v, depth + 1)? != 0;
        self.mask(if set { hi } else { lo }, depth + 1)
    }

    fn value(&mut self, value: Value, depth: usize) -> Option<u32> {
        if depth >= 96 { return None; }
        if let Some(&v) = self.memo.get(&value) { return Some(v); }
        if let Some(&(_, Input::ThreadIdInGroup(c))) = self.x.io.inputs.iter().find(|&&(v, _)| v == value) {
            return self.xyz.get(usize::from(c)).copied();
        }
        let &(b, i) = self.defs.get(&value)?;
        let inst = &self.x.p.blocks[b].insts[i];
        if self.filter && let Fields::Sop { dst: Some(Operand::Scalar { code: code @ 0..=105, dwords: 1 }), src, .. } = &inst.fields {
            if inst.write(Reg::S(*code)) != Some(value) || self.x.st.guards[b][i].is_some() { return None; }
            let a = self.operand(inst, src.first()?, depth)?;
            let result = match (inst.name(), src.len()) {
                ("s_mov_b32", 1) => a,
                ("s_not_b32", 1) => !a,
                ("s_add_i32", 2) => a.wrapping_add(self.operand(inst, &src[1], depth)?),
                ("s_sub_i32", 2) => a.wrapping_sub(self.operand(inst, &src[1], depth)?),
                ("s_lshl_b32", 2) => a.wrapping_shl(self.operand(inst, &src[1], depth)? & 31),
                _ => return None,
            };
            self.memo.insert(value, result);
            return Some(result);
        }
        let Fields::Valu(v) = &inst.fields else { return None; };
        if self.x.st.guards[b][i].is_some() || v.abs != 0 || v.neg != 0 || v.opsel != 0
            || v.neg_hi != 0 || v.opsel_hi != 0 || v.clamp || v.omod != 0
            || v.sdwa.is_some() || v.dpp.is_some() || v.carry.is_some() { return None; }
        // Only a VGPR result or the low dword of an exact lane comparison.
        let result_reg = match v.dst? {
            Operand::Vector { reg, dwords: 1 } => Reg::V(reg),
            Operand::Scalar { code, dwords: 2 } if matches!(inst.name(), "v_cmp_eq_u32" | "v_cmp_gt_u32" | "v_cmpx_gt_u32")
                || self.filter && inst.name() == "v_cmpx_gt_i32" => Reg::S(code),
            _ => return None,
        };
        if inst.write(result_reg) != Some(value) { return None; }
        let a = self.operand(inst, v.src.first()?, depth)?;
        // The filter uses floating-point arithmetic only on small exact
        // integer coordinates. Reject values where any conversion/addition
        // would require rounding, saturation or NaN handling.
        let small_int = |n: i32| (-(1 << 20)..=(1 << 20)).contains(&n);
        let small_float = |bits: u32| {
            let f = f32::from_bits(bits);
            (f.is_finite() && f.fract() == 0.0 && f.abs() <= (1 << 20) as f32).then_some(f)
        };
        if self.filter && v.src.len() == 1 {
            let result = match inst.name() {
                "v_mov_b32" => a,
                "v_cvt_f32_i32" if small_int(a as i32) => (a as i32 as f32).to_bits(),
                "v_cvt_f32_u32" if a <= 1 << 20 => (a as f32).to_bits(),
                "v_cvt_i32_f32" => small_float(a)? as i32 as u32,
                _ => return None,
            };
            self.memo.insert(value, result);
            return Some(result);
        }
        let b = self.operand(inst, v.src.get(1)?, depth)?;
        let result = match (inst.name(), v.src.len()) {
            ("v_add_nc_u32", 2) => a.wrapping_add(b),
            ("v_or_b32", 2) if self.filter => a|b,
            ("v_lshlrev_b32", 2) => b.wrapping_shl(a & 31),
            ("v_lshl_add_u32", 3) => a.wrapping_shl(b & 31).wrapping_add(self.operand(inst, &v.src[2], depth)?),
            ("v_add_lshl_u32", 3) => a.wrapping_add(b).wrapping_shl(self.operand(inst, &v.src[2], depth)? & 31),
            ("v_cmp_eq_u32", 2) => u32::from(a == b),
            ("v_cmp_gt_u32" | "v_cmpx_gt_u32", 2) => u32::from(a > b),
            ("v_cmpx_gt_i32", 2) if self.filter => u32::from((a as i32) > (b as i32)),
            ("v_add_f32", 2) if self.filter => {
                let f = small_float(a)? + small_float(b)?;
                small_float(f.to_bits())?.to_bits()
            }
            ("v_cndmask_b32", 3) => {
                let Operand::Scalar { code, dwords: 2 } = v.src[2] else { return None; };
                let mask = self.x.st.mask_bits(inst.read(Reg::S(code))?, Some(inst.read(Reg::S(code + 1))?))?;
                if self.mask(mask, depth + 1)? { b } else { a }
            }
            _ => return None,
        };
        self.memo.insert(value, result);
        Some(result)
    }
}

#[cfg(test)]
pub(crate) fn implicit_lds_audit(parts: &Parts) -> Option<usize> {
    let bad = HashMap::new();
    let l = Lower::new(parts, &bad);
    l.implicit_lds_barrier.map(|((b, i), _)| parts.p.blocks[b].insts[i].machine.offset)
}

#[cfg(test)]
pub(crate) fn implicit_lds_coverage(parts: &Parts) -> Option<(usize, usize)> {
    implicit_lds_phase(parts).map(|(_, writes, reads)| (writes, reads))
}

#[cfg(test)]
pub(crate) fn filter_lds_audit(parts: &Parts) -> Option<(usize, usize, usize)> {
    filter_lds_phase(parts).map(|((b,i), writes, reads)| (parts.p.blocks[b].insts[i].machine.offset, writes, reads))
}

#[cfg(test)]
pub(crate) fn matrix_lds_audit(parts:&Parts) -> Option<(Site,Site,usize,usize)> {
    matrix_lds_phase(parts)
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// A packed pixel input half, or `None` for padding explicitly described by
/// the metadata. With an inferred partner, a missing half instead remains an
/// error: the inference must not turn real data into zeroes.
fn packed_input_half(
    elements: &[SigElement],
    a: u8,
    b: Option<u8>,
    h: u8,
    explicit_second_row: bool,
) -> Result<Option<(u8, u8)>, Why> {
    let half = packed_half(elements, a, b, h);
    if half.is_none() && !explicit_second_row {
        Err("a packed input half no element holds")
    } else {
        Ok(half)
    }
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs packing two 16-bit rows per dword whose second row the
/// metadata leaves out (`row2` = `row`): as for outputs
/// ([`half_outputs`]), the next 16-bit row of the signature that no input
/// entry names, taken in row order (rows the metadata does give are taken
/// first).
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    partners
}

/// Loads whose eight SGPRs are used only as one image atomic's resource.
/// The atomic rebuilds the heap index from the load address.
fn atomic_descriptor_loads(p: &Program, res: &Resources) -> HashSet<Site> {
    let mut loads = HashSet::new();
    let mut readers = HashMap::new();
    for d in &res.descriptors {
        let Source::Loaded { load, first: 0 } = d.source else {
            continue;
        };
        let atomic = &p.blocks[d.site.0].insts[d.site.1];
        let Fields::Image {
            rsrc: Operand::Scalar { code, dwords: 8 },
            ..
        } = atomic.fields
        else {
            continue;
        };
        let inst = &p.blocks[load.0].insts[load.1];
        if !atomic.name().starts_with("image_atomic_")
            || inst.name() != "s_load_dwordx8"
            || code > crate::ir::VCC - 8
            || inst.writes.len() != 8
        {
            continue;
        }
        if inst.writes.iter().all(|&(reg, v)| {
            matches!(reg, Reg::S(n) if (code..code + 8).contains(&n)) && atomic.read(reg) == Some(v)
        }) {
            loads.insert(load);
            for &(_, v) in &inst.writes {
                readers.insert(v, (load, d.site));
            }
        }
    }
    if !loads.is_empty() {
        for (b, block) in p.blocks.iter().enumerate() {
            for v in block.phis.iter().flat_map(|phi| &phi.args) {
                if let Some((load, _)) = readers.get(v) {
                    loads.remove(load);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for (_, v) in &inst.reads {
                    if let Some(&(load, reader)) = readers.get(v)
                        && reader != (b, i)
                    {
                        loads.remove(&load);
                    }
                }
            }
        }
    }
    loads
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let mut console_lanes = false;
        let mut implicit_lds_barrier = None;
        let mut matrix_lds_barriers = None;
        let mut lds_allocation = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
            }) || {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                }) || insts().any(|i| {
                    matches!(
                        i.name(),
                        "v_cndmask_b32"
                            | "v_add_co_ci_u32"
                            | "v_sub_co_ci_u32"
                            | "v_subrev_co_ci_u32"
                    ) && match &i.fields {
                        Fields::Valu(v) => match v.src.get(2) {
                            Some(&Operand::Scalar { code, .. }) => i
                                .read(Reg::S(code))
                                .is_none_or(|m| !x.st.masks.contains_key(&m)),
                            _ => false,
                        },
                        _ => false,
                    }
                })
            };
            let wave = if x.native.wave64 { 64 } else { 32 };
            let one_wave = cs.threads.iter().product::<u32>() <= wave;
            if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
            }
            if !console_lanes && one_wave && s.lds_dwords > 0 {
                implicit_lds_barrier = implicit_lds_phase(x).map(|(site, _, _)| (site, 9))
                    .or_else(|| filter_lds_phase(x).map(|(site, _, _)| (site, 11)));
                lds_allocation = certify_lds_allocation(x);
                matrix_lds_barriers = matrix_lds_phase(x).map(|(publish,retire,_,_)|(publish,retire));
            }
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        let fence_only_barrier = divergent_barriers(p, x.st, &no_barrier);
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            root_sizes: HashMap::new(),
            static_table_cbvs: Vec::new(),
            static_cbv_sites: HashMap::new(),
            static_cbv_headers: HashMap::new(),
            samplers: Vec::new(),
            atomic_descriptor_loads: atomic_descriptor_loads(p, x.res),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            src0: None,
            console_lanes,
            implicit_lds_barrier,
            matrix_lds_barriers,
            lds_allocation,
            flat: None,
            mesh_broadcasts: HashMap::new(),
            logical_lane_sites: HashMap::new(),
            relative_waterfalls: relative_read_waterfalls(p, x.st),
            no_barrier,
            fence_only_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        // Root views: the dwords read checked against the view's size.
        let root = if a.param == CODE {
            None
        } else {
            self.root_view_size(&Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            })
        };
        let (off, ok) = match root {
            Some(size) => {
                let bytes = if sub.is_some() { 4 } else { 4 * n };
                let not3 = self.s.m.i32(!3);
                let at = self.s.m.bin(c, Bin::And, off, not3);
                let (ok, _) = self.in_root_view(size, at, bytes);
                let zero = self.s.m.i32(0);
                (self.s.m.select(c, ok, off, zero), Some(ok))
            }
            None => (off, None),
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            let x = self.zero_unless(ok, x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    fn run(mut self, logical_lanes: bool, static_cbvs: bool) -> Result<Lowered, Stop> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.plan_mesh_broadcasts();
        if static_cbvs {
            (self.static_table_cbvs, self.static_cbv_sites) = static_cbv_plan(self.x);
            if !self.static_table_cbvs.is_empty() {
                // The plan covers every native use, including any whose
                // value is later dead. Keep the binding and metadata in
                // agreement even when a read is never demanded as data.
                self.s.root_srv_handle(super::STATIC_CBV_SCRATCH);
            }
        }
        if logical_lanes { self.plan_logical_lane_sites(); }
        if !self.logical_lane_sites.is_empty() {
            self.mesh_thread();
        }
        if let Some(plan) = self.lds_allocation {
            if !self.allocation_plan_proven(plan) {
                self.lds_allocation = None;
            } else {
                self.s.lds_dwords += 1;
            }
        }
        self.inputs()?;
        if self.lds_allocation.is_some() {
            self.mesh_thread();
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                if self.relative_waterfalls.contains_key(&b) {
                    if phi.reg == Reg::S(EXEC) || phi.reg == Reg::S(EXEC + 1) {
                        self.vals.insert(phi.value, Val::Missing("a lane mask"));
                    } else {
                        let pre = p.blocks[b].preds.iter().position(|&q| q != b).expect("certified preheader");
                        let entry = phi.args[pre];
                        let val = self.vals.get(&entry).copied().unwrap_or(Val::Missing("a certified waterfall entry"));
                        self.vals.insert(phi.value, val);
                    }
                    continue;
                }
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            // Native waterfall SSA: the backedge phi records partial
            // completion of earlier lanes. Its entry value is the preceding
            // VGPR value; the direct indexed read replaces *all* iterations,
            // so it must consume the entry value and produce the original
            // final relative-move SSA destination.
            if let Some(plan) = self.relative_waterfalls.get(&b).copied() {
                let inst = &p.blocks[b].insts[plan.relative];
                let Fields::Valu(f) = &inst.fields else { unreachable!() };
                let Some(&Operand::Vector { reg: src, .. }) = f.src.first() else { unreachable!() };
                let Some(Operand::Vector { reg: dst, .. }) = f.dst else { unreachable!() };
                let index = self.i32_of(plan.index).map_err(|why| unsupported(inst.machine.offset, why))?;
                let value = self.rel_read(inst, src, index).map_err(|why| unsupported(inst.machine.offset, why))?;
                self.write_placed(inst, Reg::V(dst), value, None);
            }
            if let Some(plan) = self.lds_allocation {
                if b == plan.gate {
                    self.allocation_sync(9);
                    let ptr = self.allocation_abort_ptr(plan.abort_slot);
                    let flag = self.s.m.load(self.cur, ptr);
                    let zero = self.s.m.i32(0);
                    let failed = self.s.m.cmp(self.cur, pred::NE, flag, zero);
                    let (ret, go) = (self.s.m.new_block(), self.s.m.new_block());
                    self.s.m.terminate(self.cur, Term::CondBr(failed, ret, go));
                    self.s.m.terminate(ret, Term::Ret);
                    self.cur = go;
                } else if b == plan.publish {
                    self.allocation_sync(11);
                }
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                if !self.relative_waterfalls.contains_key(&b) {
                    self.inst(b, i)?;
                }
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            if self.matrix_lds_barriers.is_some_and(|(publish,_)|publish==(b,i)) {
                self.allocation_sync(11);
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        // Only relax the hardware requirement after the actual emitted code
        // confirms that no other host wave operation survived lowering.
        if self.lds_allocation.is_some() && self.s.flags & flags::WAVE_OPS != 0 {
            return Err(unsupported(0, "unexpected wave operation in certified LDS allocator"));
        }
        if !self.logical_lane_sites.is_empty() {
            if self.s.flags & flags::WAVE_OPS != 0 {
                return Err(Stop::RetryNativeLanes);
            }
            self.s.wave_size = None;
        }
        if (self.implicit_lds_barrier.is_some() || self.lds_allocation.is_some() || self.matrix_lds_barriers.is_some()) && self.s.flags & flags::WAVE_OPS == 0 {
            self.s.wave_size = None;
        }
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        #[cfg(test)]
        let op_names = self.s.m.func_names().map(str::to_owned).collect();
        #[cfg(test)]
        let insts = self.s.m.inst_lines();
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            static_table_cbvs: self.static_table_cbvs,
            heap_buffer_classes: self.s.heap_buffer_classes,
            heap_demands: self.s.heap_demands(self.user_data),
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            #[cfg(test)]
            op_names,
            #[cfg(test)]
            insts,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            // Ordinary interpolation pairs need no explicit barycentrics.
            // Materialize these inputs only if a surviving SSA use reads
            // the native I or J value as data.
            Input::Barycentric { .. } => return Val::Missing("barycentrics read directly"),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        self.materialize_barycentric(v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        Err("a value defined after its use")
    }

    fn materialize_barycentric(&mut self, v: Value) {
        if !matches!(self.vals.get(&v), Some(Val::Missing("barycentrics read directly"))) {
            return;
        }
        let Some((kind, index)) = self.x.io.inputs.iter().find_map(|&(value, input)| {
            match input {
                Input::Barycentric { kind, index } if value == v => Some((kind, index)),
                _ => None,
            }
        }) else { return };
        let value = self.barycentric_input(kind, index);
        self.vals.insert(v, value);
    }

    /// GFX10 I,J are the weights of vertices 1,2, respectively. DXIL
    /// exposes all three weights through an unpacked SV_Barycentrics
    /// input with the corresponding interpolation mode.
    fn barycentric_input(&mut self, kind: Barycentric, index: u8) -> Val {
        if self.x.native.stage != HwStage::Ps || index > 1 {
            return Val::Missing("unsupported barycentric input");
        }
        let mode = match kind {
            Barycentric::PerspectiveCenter => 2,
            Barycentric::PerspectiveCentroid => 3,
            Barycentric::PerspectiveSample => 6,
            Barycentric::LinearCenter => 4,
            Barycentric::LinearCentroid => 5,
            Barycentric::LinearSample => 7,
            Barycentric::PerspectivePullModel => return Val::Missing("pull-model barycentrics"),
        };
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self.s.inputs.iter().position(|e| bary(e) && (e.interp == mode || e.usage == 0)) {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 { return Val::Missing("barycentrics of a third interpolation mode"); }
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(), index, kind: 28, comp: 9,
                    interp: mode, row: None, col: 0, cols: 3, usage: 0, half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols != 3 || e.comp != 9 {
            return Val::Missing("invalid barycentrics signature element");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// For a root view (a root CBV, SRV or UAV), the bytes the host's view
    /// holds from its start, which the host passes in the user-data dword of
    /// the view's address. A root view has no bounds on the host: an access
    /// past what the host mapped (an offset from data the host has not
    /// mirrored) is a GPU page fault, which some hosts' GPUs stall on until
    /// the driver resets the device. Accesses are checked against this size
    /// as the console's bounded buffer views are.
    fn root_view_size(&mut self, src: &Source) -> Option<V> {
        let Source::Built { address, .. } = src else {
            return None;
        };
        if address.param == CODE {
            return None;
        }
        let rp = *self.x.native.layout.get(address.param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        if let Some(&v) = self.root_sizes.get(&address.param) {
            return Some(v);
        }
        let index = u32::from(rp.user_data);
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: RootType::Descriptor,
                dword: 0,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        let v = self.s.m.extract(entry, r, index % 4);
        self.root_sizes.insert(address.param, v);
        Some(v)
    }

    /// Whether `bytes` bytes at byte `off` lie in a root view of `size`
    /// bytes, and the offset to access instead (0 when they do not).
    fn in_root_view(&mut self, size: V, off: V, bytes: u32) -> (V, V) {
        let c = self.cur;
        let below = self.s.m.cmp(c, pred::IULT, off, size);
        let left = self.s.m.bin(c, Bin::Sub, size, off);
        let n = self.s.m.i32(bytes);
        let fits = self.s.m.cmp(c, pred::IUGE, left, n);
        let ok = self.s.m.bin(c, Bin::And, below, fits);
        let zero = self.s.m.i32(0);
        (ok, self.s.m.select(c, ok, off, zero))
    }

    /// `guard` (lanes that run an access) narrowed to `ok`.
    fn narrow(&mut self, guard: Option<V>, ok: Option<V>) -> Option<V> {
        match (guard, ok) {
            (Some(g), Some(k)) => Some(self.s.m.bin(self.cur, Bin::And, g, k)),
            (g, k) => g.or(k),
        }
    }

    /// `x` where `ok` holds, else 0 (`ok` none: always `x`).
    fn zero_unless(&mut self, ok: Option<V>, x: V) -> V {
        match ok {
            Some(k) => {
                let zero = self.s.m.i32(0);
                self.s.m.select(self.cur, k, x, zero)
            }
            None => x,
        }
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = bit?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high {
            y
        } else if self.mask_data == MaskData::Host && !self.x.native.wave64 {
            // A wave32 program's host wave may still have 64 lanes, whose
            // upper half is a second console wave: its bits are the ballot's
            // second dword.
            let lane = self.lane_index();
            let c = self.cur;
            let k32 = self.s.m.i32(32);
            let upper = self.s.m.cmp(c, pred::IUGE, lane, k32);
            self.s.flags |= flags::WAVE_OPS;
            self.s.m.select(c, upper, y, x)
        } else {
            x
        })
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as a 16-bit snorm (`signed`) or unorm in the low bits, the high
    /// bits 0: clamped to [-1, 1] or [0, 1], scaled by 32767 or 65535 and
    /// rounded to nearest even once, from the exact product. NaN gives 0.
    ///
    /// The product is formed in integers from the float's bits: in f32,
    /// `x * 32767` can need up to 39 significant bits and would round
    /// before the rounding to an integer. Denormals give 0 (below 2^-126,
    /// the scaled value is far under one half).
    fn norm16(&mut self, f: V, signed: bool) -> V {
        let c = self.cur;
        let (i32t, i64t) = (self.s.t.i32, self.s.m.int(64));
        self.s.flags |= flags::INT64;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let k64 = |l: &mut Self, v: i64| l.s.m.int_const(i64t, v);
        let max: u32 = if signed { 0x7FFF } else { 0xFFFF };
        let bits = self.as_int(f);
        let abs_mask = k(self, 0x7FFF_FFFF);
        let mag = self.s.m.bin(c, Bin::And, bits, abs_mask);
        // Exponent 1..=126 (magnitudes in [2^-126, 1)); the rest below.
        let exp = {
            let k23 = k(self, 23);
            self.s.m.bin(c, Bin::LShr, mag, k23)
        };
        let mant = {
            let m = k(self, 0x7F_FFFF);
            let one = k(self, 0x80_0000);
            let m = self.s.m.bin(c, Bin::And, mag, m);
            self.s.m.bin(c, Bin::Or, m, one)
        };
        let mant = self.s.m.cast(c, Cast::ZExt, mant, i64t);
        let scale = k64(self, i64::from(max));
        let p = self.s.m.bin(c, Bin::Mul, mant, scale);
        // The value is p / 2^s with s = 150 - exp, at least 24. From s = 41
        // on, p < 2^40 is under one half of 2^s: capping s at 63 keeps the
        // shifts defined and the result (0) the same.
        let s = {
            let k150 = k(self, 150);
            let s = self.s.m.bin(c, Bin::Sub, k150, exp);
            let k63 = k(self, 63);
            let big = self.s.m.cmp(c, pred::IUGT, s, k63);
            let s = self.s.m.select(c, big, k63, s);
            self.s.m.cast(c, Cast::ZExt, s, i64t)
        };
        let q = self.s.m.bin(c, Bin::LShr, p, s);
        let one = k64(self, 1);
        let unit = self.s.m.bin(c, Bin::Shl, one, s);
        let low = self.s.m.bin(c, Bin::Sub, unit, one);
        let rem = self.s.m.bin(c, Bin::And, p, low);
        let half = self.s.m.bin(c, Bin::LShr, unit, one);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, one);
        let zero64 = k64(self, 0);
        let odd = self.s.m.cmp(c, pred::NE, odd, zero64);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let up = self.s.m.cast(c, Cast::ZExt, up, i64t);
        let n = self.s.m.bin(c, Bin::Add, q, up);
        let n = self.s.m.cast(c, Cast::Trunc, n, i32t);
        // Magnitudes of 1 and more (and infinity) give the maximum; below
        // 2^-126 (zero, denormals) 0; NaN 0.
        let one_f = k(self, 0x3F80_0000);
        let sat = self.s.m.cmp(c, pred::IUGE, mag, one_f);
        let kmax = k(self, max);
        let n = self.s.m.select(c, sat, kmax, n);
        let zero = k(self, 0);
        let tiny = self.s.m.cmp(c, pred::EQ, exp, zero);
        let n = self.s.m.select(c, tiny, zero, n);
        let inf = k(self, 0x7F80_0000);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let n = self.s.m.select(c, nan, zero, n);
        // The sign: negated for snorm, 0 for unorm.
        let neg = self.s.m.cmp(c, pred::NE, mag, bits);
        if signed {
            let minus = self.s.m.bin(c, Bin::Sub, zero, n);
            let r = self.s.m.select(c, neg, minus, n);
            let ffff = k(self, 0xFFFF);
            self.s.m.bin(c, Bin::And, r, ffff)
        } else {
            self.s.m.select(c, neg, zero, n)
        }
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9. As on RDNA2, a denormal operand counts
    /// as zero, so the test is on the exponent bits rather than a float
    /// compare (whose result would depend on the host's denormal mode).
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let exp = self.s.m.i32(0x7F80_0000);
        let izero = self.s.m.i32(0);
        let is_zero = |l: &mut Self, v: V| {
            let bits = l.as_int(v);
            let e = l.s.m.bin(c, Bin::And, bits, exp);
            l.s.m.cmp(c, pred::EQ, e, izero)
        };
        let az = is_zero(self, a);
        let bz = is_zero(self, b);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        let zero = self.s.m.f32(0);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit from a mask operand (`v_cndmask_b32` / `*_ci_u32`):
    /// uses the tracked boolean when `structure` followed it, handles inline
    /// `0`/`-1` constants, or tests this console lane's bit of the scalar
    /// mask register (pair in wave64) when console lanes are active.
    fn mask_cond(
        &mut self,
        inst: &Inst,
        op: Option<&Operand>,
        at: usize,
        err: Why,
    ) -> Result<V, Why> {
        match op {
            Some(&Operand::Const(128)) => Ok(self.s.m.bool(false)),
            Some(&Operand::Const(193)) => Ok(self.s.m.bool(true)),
            Some(&Operand::Scalar { code, .. }) => {
                if let Some(m) = inst
                    .read(Reg::S(code))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                {
                    return self.boolean(m, at);
                }
                if !self.console_lanes {
                    return Err(err);
                }
                let lo = inst.read(Reg::S(code)).ok_or(err)?;
                let lo = self.i32_of(lo)?;
                let hi = if self.x.native.wave64 {
                    match inst.read(Reg::S(code + 1)) {
                        Some(v) => Some(self.i32_of(v)?),
                        None => Some(self.s.m.i32(0)),
                    }
                } else {
                    None
                };
                self.s.flags |= flags::WAVE_OPS;
                Ok(self.lane_bit(at, lo, hi))
            }
            _ => Err(err),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The two register offsets of `v_movrelsd_2_b32` and `v_swaprel_b32`:
    /// M0[9:0] for the source, M0[25:16] for the destination.
    fn m0_halves(&mut self, m0: V) -> (V, V) {
        let c = self.cur;
        let low = self.s.m.i32(0x3FF);
        let sixteen = self.s.m.i32(16);
        let si = self.s.m.bin(c, Bin::And, m0, low);
        let high = self.s.m.bin(c, Bin::LShr, m0, sixteen);
        let di = self.s.m.bin(c, Bin::And, high, low);
        (si, di)
    }

    /// `v[base + index]`, from the M0-relative range `inst` reads from
    /// `base` on ([`crate::ir`] names it up to the program's VGPR count);
    /// 0 past it.
    fn rel_read(&mut self, inst: &Inst, base: u8, index: V) -> Result<V, Why> {
        let c = self.cur;
        let mut r = self.s.m.i32(0);
        for k in 0..=255u8.saturating_sub(base) {
            let Some(x) = inst.read(Reg::V(base + k)) else {
                break;
            };
            let x = self.i32_of(x)?;
            let kk = self.s.m.i32(u32::from(k));
            let is = self.s.m.cmp(c, pred::EQ, index, kk);
            r = self.s.m.select(c, is, x, r);
        }
        Ok(r)
    }

    /// The value a VGPR an M0-relative write may land on held before it
    /// (0 when the program never set it).
    fn old_vgpr(&mut self, inst: &Inst, reg: u8) -> Result<V, Why> {
        match inst.read(Reg::V(reg)) {
            Some(o) => self.i32_of(o),
            None => Ok(self.s.m.i32(0)),
        }
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// The lane's own console wave's 32-lane base on a 64-lane host wave
    /// (0 or 32): the renumbering a console lane number takes to stay a
    /// host lane.
    fn lane_base(&mut self) -> V {
        let lane = self.lane_index();
        let k32 = self.s.m.i32(32);
        self.s.m.bin(self.cur, Bin::And, lane, k32)
    }

    /// A lane operand as a host lane: a constant one is the console's, so
    /// on a host wave wider than the program's each half is a console wave
    /// and console lane `k` of the lane's own wave is `k | (lane & 32)`.
    /// On a wave64 program lanes already match one to one.
    fn console_lane(&mut self, lane: V) -> V {
        if matches!(lane, V::Const(_)) && !self.x.native.wave64 {
            let base = self.lane_base();
            self.s.m.bin(self.cur, Bin::Or, lane, base)
        } else {
            lane
        }
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes || !self.logical_lane_sites.is_empty() {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let z = self.s.m.i32(0);
        let passes = mesh.max_primitives.div_ceil(threads.max(1));
        for pass in 0..passes {
            let idx = if pass == 0 {
                flat
            } else {
                let off = self.s.m.i32(pass * threads);
                self.s.m.bin(self.cur, Bin::Add, flat, off)
            };
            let need_guard = (pass + 1) * threads > mesh.max_primitives;
            let guard = need_guard.then(|| self.s.m.cmp(self.cur, pred::IULT, idx, p));
            let at = self.begin_guarded(guard);
            self.s.op(
                self.cur,
                "emitIndices",
                Attr::NoUnwind,
                void,
                169,
                &[idx, z, z, z],
            );
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// A deliberately restricted group rendezvous: no native LDS, no
    /// partial EXEC, no simulated whole-wave sections, no loops on the
    /// path to the rendezvous, and no possible revisit afterwards.
    fn plan_mesh_broadcasts(&mut self) {
        let StageIo::Ms(ms) = &self.x.native.io else { return };
        if !self.x.native.wave64 || ms.lds_bytes != 0
            || ms.threads == 0 || ms.threads > 128 || ms.threads % 64 != 0 {
            return;
        }
        let waves = ms.threads / 64;
        for (b, block) in self.x.p.blocks.iter().enumerate() {
            if !once_only_rendezvous(&self.x.st.exits, b) { continue; }
            for (i, inst) in block.insts.iter().enumerate() {
                if inst.name() != "v_readfirstlane_b32"
                    || self.x.st.guards[b][i].is_some()
                    || !self.full_exec_at(inst)
                    || self.sections.get(&b).is_some_and(|ss|
                        ss.iter().any(|(&start, sec)| start <= i && i < sec.end))
                    || inst.writes.first().is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) {
                    continue;
                }
                let Fields::Valu(v) = &inst.fields else { continue };
                if !matches!(v.src.first(), Some(Operand::Vector { dwords: 1, .. }))
                    || !matches!(v.dst, Some(Operand::Scalar { code: 0..=105, dwords: 1 }))
                    || v.abs != 0 || v.neg != 0 || v.opsel != 0 || v.sdwa.is_some() || v.dpp.is_some() {
                    continue;
                }
                if inst.read(Reg::V(5)).is_some_and(|value| self.x.io.inputs.iter()
                    .any(|&(x, input)| x == value && input == Input::MeshGroup))
                    || self.uniform_scalar_select_at_firstlane(inst)
                    || self.uniform_buffer_fetch_at_firstlane(inst)
                    || self.uniform_arithmetic_firstlane(inst, &mut HashSet::new(), 0) {
                    continue;
                }
                let base = self.s.lds_dwords;
                self.s.lds_dwords += waves;
                self.mesh_broadcasts.insert((b, i), base);
            }
        }
    }

    fn mesh_broadcast(&mut self, base: u32, value: V) -> V {
        let flat = self.mesh_thread();
        let c = self.cur;
        let (six, mask, zero, base) = (self.s.m.i32(6), self.s.m.i32(63), self.s.m.i32(0), self.s.m.i32(base));
        let wave = self.s.m.bin(c, Bin::LShr, flat, six);
        let lane = self.s.m.bin(c, Bin::And, flat, mask);
        let index = self.s.m.bin(c, Bin::Add, base, wave);
        let lds = self.s.lds();
        let ptr = self.s.m.element(c, lds, index);
        let leader = self.s.m.cmp(c, pred::EQ, lane, zero);
        let at = self.begin_guarded(Some(leader));
        self.s.m.store(self.cur, ptr, value);
        self.end_guarded(at, &[]);
        // The join is reached by every group thread, not just the leader.
        // Each site has disjoint storage and executes once, so no reuse fence.
        let mode = self.s.m.i32(9);
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[mode]);
        self.s.m.load(self.cur, ptr)
    }

    /// An entire native wave must converge at each snapshot. In the original
    /// straight-line case a source SSA value may reuse its publication. In a
    /// CFG, each publication gets distinct storage; prove all paths from the
    /// entry reach it once, full EXEC, and no later path revisits it. This
    /// admits mip reduction phases separated by guarded output stores, but
    /// not a reduction inside a divergent loop or partial-EXEC region.
    // Native LDS combined with whole-wave scans requires a separate proof
    // of both scan convergence and LDS visibility (including packed writes).
    // Keep the native WaveSize requirement until that proof exists.
    fn plan_logical_lane_sites(&mut self) {
        let StageIo::Cs(cs) = self.x.native.io else { return };
        let p = self.x.p;
        let st = self.x.st;
        if !self.x.native.wave64 || !p.wave64 || cs.lds_bytes() != 0
            || cs.threads.iter().product::<u32>() != 64
            || !self.sections.is_empty() {
            return;
        }
        let straight = p.blocks.len() == 1;
        if straight && (!matches!(p.blocks[0].exit, Exit::End) || st.exits.as_slice() != [LaneExit::End]) { return; }
        // In a multi-block shader, do not move inter-lane visibility of a
        // UAV store across a subsequent data read. Only a pure input phase
        // followed by output stores is covered. Scalar descriptor loads
        // may remain between output phases; a scalar buffer data load may
        // not. Reject atomics, which can publish data during the wave.
        if !straight {
            let mut after_store = false;
            let order = crate::ir::reverse_postorder(&p.blocks);
            let rank: HashMap<_,_> = order.iter().enumerate().map(|(n,&b)|(b,n)).collect();
            // RPO is a topological order only in an acyclic CFG. Refuse
            // backedges even if they occur after the final rendezvous.
            if order.len() != p.blocks.len() || order.iter().any(|&b| {
                p.blocks[b].exit.successors().iter().any(|t| rank.get(t).is_none_or(|&r| r <= rank[&b]))
            }) { return; }
            for b in order {
                for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                    let n = inst.name();
                    if n.contains("atomic") || n == "s_barrier" || n.starts_with("s_sendmsg")
                        || matches!(inst.fields, Fields::Flat { .. }) && !n.starts_with("global_load_") { return; }
                    if after_store && (matches!(inst.fields, Fields::Image { .. } | Fields::Buffer { .. } | Fields::Flat { .. }) && !n.contains("store")
                        || n.starts_with("s_buffer_load_") || n.starts_with("global_load_")) { return; }
                    if after_store && matches!(inst.fields, Fields::Smem { .. }) && !self.x.res.scalar.iter().any(|&(site, a)| {
                        site == (b,i) && (a.param == CODE || self.x.native.layout.get(a.param).is_some_and(|r| r.root == RootType::Table))
                    }) { return; }
                    if n.contains("store") { after_store = true; }
                }
            }
        }
        let mut sites = HashMap::new();
        let mut sources = HashMap::new();
        let mut slots = 0u32;
        for (b, bb) in p.blocks.iter().enumerate() { for (i, inst) in bb.insts.iter().enumerate() {
            if (straight && (st.guards[b][i].is_some() || inst.writes.iter().any(|&(r, _)| matches!(r, Reg::S(EXEC) | Reg::S(127)))))
                || reads_mask_as_data(st, inst)
                || inst.name() == "s_get_waveid_in_workgroup" {
                return;
            }
            if straight && inst.read(Reg::S(EXEC)).is_some() && !self.full_exec_at(inst) { return; }
            let reg = match &inst.fields {
                Fields::Ds { addr: Some(Operand::Vector { reg, dwords: 1 }),
                    dst: Some(Operand::Vector { dwords: 1, .. }), offset0, offset1, gds: false, .. }
                    if inst.name() == "ds_swizzle_b32"
                        && (u16::from(*offset0) | u16::from(*offset1) << 8) < 0x8000 => *reg,
                Fields::Ds { .. } => return,
                Fields::Valu(v) if inst.name() == "v_readlane_b32" => {
                    let [Operand::Vector { reg, dwords: 1 }, lane] = v.src.as_slice() else { return };
                    let lane = match lane {
                        Operand::Const(n) => crate::resource::inline(*n),
                        Operand::Literal(n) => Some(*n),
                        _ => None,
                    };
                    if !lane.is_some_and(|n| n < 64)
                        || !matches!(v.dst, Some(Operand::Scalar { code: 0..=105, dwords: 1 }))
                        || v.dpp.is_some() || v.sdwa.is_some() || v.abs != 0 || v.neg != 0
                        || v.opsel != 0 || v.clamp || v.omod != 0 || v.carry.is_some() { return; }
                    *reg
                }
                Fields::Valu(v) if !straight && matches!(inst.name(), "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32") => {
                    // Constant all-ones masks are lane arithmetic, with no
                    // actual ballot. Existing mbcnt lowering operates on
                    // the native flattened thread id in this certified plan.
                    if v.src.first() != Some(&Operand::Const(193)) || v.dpp.is_some() || v.sdwa.is_some()
                        || v.abs != 0 || v.neg != 0 || v.opsel != 0 || v.clamp || v.omod != 0 || v.carry.is_some() { return; }
                    continue;
                }
                Fields::Valu(v) if v.dpp.is_some() || inst.name().contains("lane")
                    || inst.name().contains("permute") || inst.name().starts_with("v_mbcnt") => return,
                _ => continue,
            };
            if st.guards[b][i].is_some() || inst.writes.iter().any(|&(r, _)| matches!(r, Reg::S(EXEC) | Reg::S(127)))
                || inst.read(Reg::S(EXEC)).is_some() && !self.full_exec_at(inst)
                || !straight && (st.reach[b] != Bool::TRUE || !once_only_rendezvous(&st.exits,b)) { return; }
            let Some(value) = inst.read(Reg::V(reg)) else { return };
            let (base, publish) = if straight && let Some(&base) = sources.get(&value) {
                (base, false)
            } else {
                let base = slots * 64;
                if base + 64 > 8192 { return; } // D3D12's 32 KiB group limit
                sources.insert(value, base);
                slots += 1;
                (base, true)
            };
            sites.insert((b, i), (base, publish));
        } }
        if !sites.is_empty() {
            self.s.lds_dwords = slots * 64;
            self.logical_lane_sites = sites;
        }
    }

    fn logical_lane_read(&mut self, base: u32, publish: bool, value: V, src: V) -> V {
        let flat = self.mesh_thread();
        let c = self.cur;
        let base = self.s.m.i32(base);
        let lds = self.s.lds();
        if publish {
            let index = self.s.m.bin(c, Bin::Add, base, flat);
            let ptr = self.s.m.element(c, lds, index);
            self.s.m.store(c, ptr, value);
            let mode = self.s.m.i32(if self.x.p.blocks.len() > 1 && self.writes_uavs { 11 } else { 9 });
            self.s.op(c, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[mode]);
        }
        let index = self.s.m.bin(c, Bin::Add, base, src);
        let ptr = self.s.m.element(c, lds, index);
        self.s.m.load(c, ptr)
    }

    /// `v_cndmask` selecting two scalar values under a scalar (SCC-only)
    /// predicate has the same output in every executing lane. This proves a
    /// following `v_readfirstlane` is a no-op only when its EXEC is exactly
    /// the writer's EXEC: inactive lanes may retain a different old VGPR.
    /// A lane compare, phi, opaque/constant partial mask, or intervening
    /// EXEC change is deliberately not accepted.
    fn uniform_scalar_select_at_firstlane(&self, read: &Inst) -> bool {
        self.uniform_scalar_select_proven(read, &mut HashSet::new(), 0)
    }

    /// Recursively prove the scalar values used by a source select rather
    /// than assuming any SGPR or SCC value is safe: scalar registers may
    /// have received a previous *unproven* cross-lane VGPR read.
    fn scalar_value_proven(
        &self, value: Value, visiting: &mut HashSet<Value>, depth: usize,
    ) -> bool {
        self.scalar_value_proven_impl(value, visiting, depth, false)
    }

    // A partial fetch may trace a scalar through a single uniform diamond
    // inside its active region. Keep that local proof out of global scalar
    // provenance, which requires uniform incoming reach across the wave.
    fn scalar_value_proven_impl(
        &self, value: Value, visiting: &mut HashSet<Value>, depth: usize, local: bool,
    ) -> bool {
        if depth >= 96 || !visiting.insert(value) { return false; }
        let safe = if self.x.p.inputs.iter().any(|&(r, v)| v == value
            && matches!(r, Reg::S(_) | Reg::Scc)
            && (!local || matches!(r, Reg::S(0..=105) | Reg::Scc))) {
            true // Hardware scalar system inputs are per-console-wave values.
        } else if let Some((block, phi)) = self.x.p.blocks.iter().enumerate()
            .find_map(|(i, b)| b.phis.iter().find(|p| p.value == value).map(|p| (i, p))) {
            // A scalar phi is uniform only if every incoming scalar is
            // independently proven uniform AND its predecessor is reached
            // through a uniform control path. Different lanes taking
            // different predecessors must not be collapsed into one SGPR.
            let target = &self.x.p.blocks[block];
            matches!(phi.reg, Reg::S(_) | Reg::Scc)
                && target.preds.len() == phi.args.len()
                && !target.preds.is_empty()
                && (target.preds.iter().all(|&pred| {
                    self.x.st.bools.support(self.x.st.reach[pred]).into_iter().all(|atom| {
                        matches!(atom, Atom::Uniform(v)
                            if self.scalar_value_proven_impl(v, visiting, depth + 1, local))
                    })
                }) || (local && self.local_uniform_diamond(block, visiting, depth + 1)))
                && phi.args.iter().all(|&v| self.scalar_value_proven_impl(v, visiting, depth + 1, local))
        } else if let Some((b, i, inst)) = self.x.p.blocks.iter().enumerate()
            .flat_map(|(b, block)| block.insts.iter().enumerate().map(move |(i, inst)| (b, i, inst)))
            .find(|(_, _, inst)| inst.writes.iter().any(|&(_, w)| w == value)) {
            if inst.name() == "v_readfirstlane_b32" && self.full_exec_at(inst) {
                // A waterfall read is deliberately lowered to each lane's own
                // descriptor, not a logical-wave-uniform scalar.
                let waterfall = self.x.res.waterfall.reads.contains(&value);
                // Only the exact SSA output of a certified once-only group
                // rendezvous is uniform across its original 64-lane wave.
                let broadcast = self.mesh_broadcasts.contains_key(&(b, i))
                    && self.x.st.guards[b][i].is_none()
                    && inst.writes.first().is_some_and(|&(r, w)| w == value && matches!(r, Reg::S(0..=105)));
                let group = matches!(&inst.fields,
                    Fields::Valu(v) if matches!(v.src.first(), Some(Operand::Vector { reg: 5, .. }))
                        && inst.read(Reg::V(5)).is_some_and(|x| self.x.io.inputs.iter()
                            .any(|&(v, input)| x == v && input == crate::io::Input::MeshGroup)));
                !waterfall && (broadcast || group || self.uniform_scalar_select_proven(inst, visiting, depth + 1)
                    || self.uniform_buffer_fetch_proven(inst, visiting, depth + 1)
                    || self.uniform_arithmetic_firstlane(inst, visiting, depth + 1))
            } else if matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. })
                && inst.name().starts_with("s_")
                && !inst.name().contains("readlane")
                && !inst.name().contains("bcnt1")
                && !inst.name().contains("ff1") {
                inst.reads.iter().all(|&(r, v)| {
                    matches!(r, Reg::S(_) | Reg::Scc)
                        && (!local || matches!(r, Reg::S(0..=105) | Reg::Scc))
                        && self.scalar_value_proven_impl(v, visiting, depth + 1, local)
                })
            } else { false }
        } else { false };
        visiting.remove(&value);
        safe
    }

    /// Exactly one conditional edge goes directly to the join; the other
    /// goes through a single-predecessor arm. All lanes reaching the head
    /// agree on the simplified branch (even inside a partial EXEC region).
    fn local_uniform_diamond(&self, join: usize, visiting: &mut HashSet<Value>, depth: usize) -> bool {
        if depth >= 96 { return false; }
        let blocks = &self.x.p.blocks;
        let exits = &self.x.st.exits;
        if exits.iter().enumerate().any(|(b, e)| match *e {
            LaneExit::Jump(t) => t <= b,
            LaneExit::Branch { taken, not_taken, .. } => taken <= b || not_taken <= b,
            LaneExit::Indirect => true,
            LaneExit::End => false,
        }) { return false; }
        let [a, b] = blocks[join].preds[..] else { return false; };
        [(a, b), (b, a)].into_iter().any(|(head, arm)| {
            if blocks[arm].preds.as_slice() != [head]
                || exits[arm] != LaneExit::Jump(join) { return false; }
            let LaneExit::Branch { when, taken, not_taken } = exits[head] else { return false; };
            if !((taken == join && not_taken == arm) || (taken == arm && not_taken == join)) {
                return false;
            }
            self.x.st.bools.support(when).into_iter().all(|atom| matches!(atom,
                Atom::Uniform(v) if self.scalar_value_proven_impl(v, visiting, depth + 1, true)))
        })
    }

    /// Pure scalar-derived VALU chains used for reciprocal division setup.
    /// Follow the actual SSA operands (not the implicit old destination).
    /// Require the same full EXEC at every definition, and reject phis,
    /// lane-dependent operations, modifiers and unknown scalar origins.
    fn uniform_arithmetic_firstlane(
        &self, read: &Inst, visiting: &mut HashSet<Value>, depth: usize,
    ) -> bool {
        if depth >= 96 || !self.full_exec_at(read) { return false; }
        let Fields::Valu(rv) = &read.fields else { return false; };
        let Some(Operand::Vector { reg, dwords: 1 }) = rv.src.first() else { return false; };
        read.read(Reg::V(*reg)).is_some_and(|value|
            self.uniform_arithmetic_value(value, read, visiting, depth + 1))
    }

    fn uniform_arithmetic_value(
        &self, value: Value, read: &Inst, visiting: &mut HashSet<Value>, depth: usize,
    ) -> bool {
        if depth >= 96 || !visiting.insert(value) { return false; }
        let safe = (|| {
            let Some((b, i, writer)) = self.x.p.blocks.iter().enumerate()
                .flat_map(|(b, block)| block.insts.iter().enumerate().map(move |(i, w)| (b, i, w)))
                .find(|(_, _, w)| w.writes.iter().any(|&(r, v)| matches!(r, Reg::V(_)) && v == value))
                else { return false; };
            let arity = match writer.name() {
                "v_cvt_f32_ubyte0" | "v_cvt_f32_u32" | "v_rcp_iflag_f32" | "v_cvt_u32_f32" => 1,
                "v_mul_legacy_f32" => 2,
                _ => return false,
            };
            let Fields::Valu(wv) = &writer.fields else { return false; };
            if self.x.st.guards[b][i].is_some()
                || !self.full_exec_at(writer)
                || writer.read(Reg::S(EXEC)) != read.read(Reg::S(EXEC))
                || writer.read(Reg::S(EXEC + 1)) != read.read(Reg::S(EXEC + 1))
                || wv.src.len() != arity || wv.abs != 0 || wv.neg != 0 || wv.opsel != 0
                || wv.neg_hi != 0 || wv.opsel_hi != 0 || wv.clamp || wv.omod != 0
                || wv.sdwa.is_some() || wv.dpp.is_some() {
                return false;
            }
            wv.src.iter().all(|op| match *op {
                Operand::Const(128..=208 | 240..=248) | Operand::Literal(_) => true,
                Operand::Scalar { code: code @ 0..=105, dwords: 1 } => writer.read(Reg::S(code))
                    .is_some_and(|v| self.scalar_value_proven(v, visiting, depth + 1)),
                Operand::Vector { reg, dwords: 1 } => writer.read(Reg::V(reg))
                    .is_some_and(|v| self.uniform_arithmetic_value(v, read, visiting, depth + 1)),
                _ => false,
            })
        })();
        visiting.remove(&value);
        safe
    }

    /// The masked division fallback uses the same scalar divisor in every
    /// participating console lane. Certify only the original reciprocal
    /// sequence, wholly inside one active block. Its SGPR result is local
    /// to that region; never feed it into global scalar provenance.
    fn partial_reciprocal_firstlane(&self, read: &Inst) -> bool {
        if self.s.mesh.is_none() || !self.x.native.wave64 || self.full_exec_at(read)
            || read.name() != "v_readfirstlane_b32" { return false; }
        let Some((b, i)) = self.x.p.blocks.iter().enumerate().find_map(|(b, bb)|
            bb.insts.iter().position(|w| std::ptr::eq(w, read)).map(|i| (b, i))) else { return false; };
        let Some(lo) = read.read(Reg::S(EXEC)) else { return false; };
        let Some(hi) = read.read(Reg::S(EXEC + 1)) else { return false; };
        let Some(mask) = self.x.st.mask_bits(lo, Some(hi)) else { return false; };
        // All lanes entering the block are active. No whole-wave emulation
        // or extra instruction guard may introduce another active set.
        if mask == Bool::FALSE || self.x.st.reach[b] == Bool::FALSE
            || !self.sections.get(&b).is_none_or(|s| s.is_empty())
            || self.x.st.guards[b][i].is_some() { return false; }
        let Fields::Valu(rv) = &read.fields else { return false; };
        let plain = |v: &crate::gfx10::Valu| v.abs == 0 && v.neg == 0 && v.opsel == 0
            && v.neg_hi == 0 && v.opsel_hi == 0 && !v.clamp && v.omod == 0
            && v.sdwa.is_none() && v.dpp.is_none() && v.carry.is_none();
        let [Operand::Vector { reg, dwords: 1 }] = rv.src.as_slice() else { return false; };
        let Some(Operand::Scalar { code: dst @ 0..=105, dwords: 1 }) = rv.dst else { return false; };
        let Some(result) = read.write(Reg::S(dst)) else { return false; };
        let Some(mut value) = read.read(Reg::V(*reg)) else { return false; };
        if !plain(rv) || read.writes.len() != 1 || self.x.res.waterfall.reads.contains(&result) {
            return false;
        }
        let block = &self.x.p.blocks[b];
        let mut before = i;
        // Work backwards through exact SSA writers and require the native
        // reciprocal scale bit pattern. No phi, move, or alternate formula.
        for name in ["v_cvt_u32_f32", "v_mul_legacy_f32", "v_rcp_iflag_f32", "v_cvt_f32_u32"] {
            let Some((j, w)) = block.insts[..before].iter().enumerate().rev()
                .find(|(_, w)| w.writes.iter().any(|&(_, v)| v == value)) else { return false; };
            let Fields::Valu(v) = &w.fields else { return false; };
            let Some(Operand::Vector { reg, dwords: 1 }) = v.dst else { return false; };
            if w.name() != name || !plain(v) || w.write(Reg::V(reg)) != Some(value)
                || w.writes.len() != 1 || self.x.st.guards[b][j].is_some()
                || w.read(Reg::S(EXEC)) != Some(lo) || w.read(Reg::S(EXEC + 1)) != Some(hi) {
                return false;
            }
            before = j;
            match (name, v.src.as_slice()) {
                ("v_cvt_f32_u32", [Operand::Scalar { code: code @ 0..=105, dwords: 1 }]) => {
                    if !w.read(Reg::S(*code)).is_some_and(|v|
                        self.scalar_value_proven(v, &mut HashSet::new(), 0)) { return false; }
                }
                ("v_mul_legacy_f32", [Operand::Literal(0x4f7ffffe), Operand::Vector { reg, dwords: 1 }])
                | ("v_mul_legacy_f32", [Operand::Vector { reg, dwords: 1 }, Operand::Literal(0x4f7ffffe)])
                | ("v_cvt_u32_f32" | "v_rcp_iflag_f32", [Operand::Vector { reg, dwords: 1 }]) => {
                    let Some(src) = w.read(Reg::V(*reg)) else { return false; };
                    value = src;
                }
                _ => return false,
            }
        }
        self.partial_reciprocal_local_uses(b, i, lo, hi, result, 0)
    }

    /// The reciprocal is consumed by a scalar correction sequence and then
    /// a per-lane multiply. Track every scalar result (including SCC) until
    /// that terminal vector use. Reject any cross-block or mask/data escape.
    fn partial_reciprocal_local_uses(&self, block: usize, after: usize, lo: Value, hi: Value,
        value: Value, depth: usize) -> bool {
        if depth >= 16 { return false; }
        for (b, bb) in self.x.p.blocks.iter().enumerate() {
            if bb.phis.iter().any(|p| p.args.contains(&value)) { return false; }
            let masks = self.x.st.guards[b].iter().flatten().copied().chain(
                match self.x.st.exits[b] { LaneExit::Branch { when, .. } => Some(when), _ => None });
            if masks.into_iter().any(|m| self.x.st.bools.support(m).into_iter().any(|a|
                matches!(a, Atom::Uniform(v) | Atom::Lane(v) | Atom::Opaque(v) if v == value))) { return false; }
            for (i, w) in bb.insts.iter().enumerate() {
                if !w.reads.iter().any(|&(_, v)| v == value) { continue; }
                if b != block || i <= after || self.x.st.guards[b][i].is_some() { return false; }
                match (&w.fields, w.name()) {
                    (Fields::Sop { dst: Some(Operand::Scalar { code: 0..=105, dwords: 1 }), src, .. },
                        "s_mul_i32" | "s_mul_hi_u32" | "s_add_i32")
                        if src.len() == 2 && src.iter().all(|op| matches!(op,
                            Operand::Scalar { code: 0..=105, dwords: 1 } | Operand::Const(128..=208) | Operand::Literal(_)))
                            && w.writes.iter().all(|&(r, v)| matches!(r, Reg::S(0..=105) | Reg::Scc)
                                && self.partial_reciprocal_local_uses(block, i, lo, hi, v, depth + 1)) => {},
                    (Fields::Valu(v), "v_mul_hi_u32") if w.read(Reg::S(EXEC)) == Some(lo)
                        && w.read(Reg::S(EXEC + 1)) == Some(hi)
                        && v.src.len() == 2 && v.abs == 0 && v.neg == 0 && v.opsel == 0
                        && v.neg_hi == 0 && v.opsel_hi == 0 && !v.clamp && v.omod == 0
                        && v.sdwa.is_none() && v.dpp.is_none() && v.carry.is_none()
                        && matches!(v.dst, Some(Operand::Vector { dwords: 1, .. }))
                        && w.writes.iter().all(|&(r, _)| matches!(r, Reg::V(_))) => {},
                    _ => return false,
                }
            }
        }
        true
    }

    /// The masked mesh attribute-table lookup converts a scalar float to
    /// an integer index, then reads three coefficients through SMEM. Only
    /// substitute the current lane when the actual SSA source is uniform,
    /// the address chain stays local, and the read is proven data rather
    /// than a descriptor or a mask. Keep this out of global scalar proof.
    fn partial_data_conversion_firstlane(&self, read: &Inst) -> bool {
        self.partial_data_conversion_site(read).is_some()
    }

    fn partial_data_conversion_site(&self, read: &Inst) -> Option<()> {
        if self.s.mesh.is_none() || !self.x.native.wave64 || self.full_exec_at(read)
            || read.name() != "v_readfirstlane_b32" { return None; }
        let (b, i) = self.x.p.blocks.iter().enumerate().find_map(|(b, bb)|
            bb.insts.iter().position(|w| std::ptr::eq(w, read)).map(|i| (b, i)))?;
        let lo = read.read(Reg::S(EXEC))?;
        let hi = read.read(Reg::S(EXEC + 1))?;
        if self.x.st.mask_bits(lo, Some(hi))? == Bool::FALSE
            || self.x.st.reach[b] == Bool::FALSE || self.x.st.guards[b][i].is_some()
            || !self.sections.get(&b).is_none_or(|s| s.is_empty()) { return None; }
        // The duplication of a scalar read on individual lanes is only
        // justified for immutable memory throughout this shader.
        if self.x.p.blocks.iter().flat_map(|bb| &bb.insts).any(|w| {
            let name = w.name();
            name.contains("store") || name.contains("atomic") || name.starts_with("ds_write")
                || name.starts_with("ds_add") || name.starts_with("ds_sub")
                || matches!(w.fields, Fields::Ds { .. } | Fields::Flat { .. })
        }) { return None; }
        let plain = |v: &crate::gfx10::Valu| v.abs == 0 && v.neg == 0 && v.opsel == 0
            && v.neg_hi == 0 && v.opsel_hi == 0 && !v.clamp && v.omod == 0
            && v.sdwa.is_none() && v.dpp.is_none() && v.carry.is_none();
        let same_exec = |w: &Inst| w.read(Reg::S(EXEC)) == Some(lo)
            && w.read(Reg::S(EXEC + 1)) == Some(hi);
        let Fields::Valu(rv) = &read.fields else { return None; };
        let [Operand::Vector { reg, dwords: 1 }] = rv.src.as_slice() else { return None; };
        let Some(Operand::Scalar { code: dst @ 0..=105, dwords: 1 }) = rv.dst else { return None; };
        let result = read.write(Reg::S(dst))?;
        if !plain(rv) || read.writes.len() != 1 || self.x.res.waterfall.reads.contains(&result) { return None; }
        let bb = &self.x.p.blocks[b];
        let vector = read.read(Reg::V(*reg))?;
        let (cv_i, cv) = bb.insts[..i].iter().enumerate().rev()
            .find(|(_, w)| w.write(Reg::V(*reg)) == Some(vector))?;
        let Fields::Valu(v) = &cv.fields else { return None; };
        let [Operand::Scalar { code: input @ 0..=105, dwords: 1 }] = v.src.as_slice() else { return None; };
        if cv.name() != "v_cvt_u32_f32" || !plain(v) || cv.writes.len() != 1
            || !matches!(v.dst, Some(Operand::Vector { reg: r, dwords: 1 }) if r == *reg)
            || !same_exec(cv) || self.x.st.guards[b][cv_i].is_some() { return None; }
        let source = cv.read(Reg::S(*input))?;
        if !self.scalar_value_proven(source, &mut HashSet::new(), 0) { return None; }
        // Require the scalar float itself to be a fixed-offset CBV data read,
        // not an inferred descriptor field or a value from a lane phi.
        let (sb, si, sw) = self.x.p.blocks.iter().enumerate().flat_map(|(b, bb)|
            bb.insts.iter().enumerate().map(move |(i, w)| (b, i, w)))
            .find(|(_, _, w)| w.writes.iter().any(|&(_, val)| val == source))?;
        let Fields::Smem { data: Some(Operand::Scalar { code: sc, dwords }),
            offset, soffset: Operand::Scalar { code: NULL, .. }, .. } = &sw.fields else { return None; };
        let component = match (sw.name(), *dwords, *offset) {
            ("s_buffer_load_dword", 1, 36 | 40 | 44) => 0,
            // The four-dword CBV variant reads the index from byte 8.
            // Certify only that exact SSA component, not the other words.
            ("s_buffer_load_dwordx4", 4, 0) if *sc <= 102 => 2,
            _ => return None,
        };
        if sw.writes.len() != usize::from(*dwords) || sw.write(Reg::S(sc.checked_add(component)?)) != Some(source)
            || self.static_readonly_scalar_descriptor((sb, si), sw)?.kind != ResourceType::Cbv { return None; }
        // Follow a single local use of each address SSA value: index * 12,
        // optional +48 (the other coefficient), plus record base.
        let uses = |after, val| self.partial_data_local_uses(b, after, val);
        let sole_use = |after, val| -> Option<usize> {
            let sites = uses(after, val)?;
            let [j] = sites.as_slice() else { return None; };
            Some(*j)
        };
        let scalar_result = |j: usize| -> Option<(u8, Value)> {
            let w = &bb.insts[j];
            let Fields::Sop { dst: Some(Operand::Scalar { code: r @ 0..=105, dwords: 1 }), .. } = &w.fields else { return None; };
            for &(reg, val) in &w.writes {
                if reg == Reg::Scc {
                    if !uses(j, val)?.is_empty() { return None; }
                } else if reg != Reg::S(*r) { return None; }
            }
            Some((*r, w.write(Reg::S(*r))?))
        };
        let mul_i = sole_use(i, result)?;
        let mul = &bb.insts[mul_i];
        let Fields::Sop { src, .. } = &mul.fields else { return None; };
        if mul.name() != "s_mul_i32" || !matches!(src.as_slice(),
            [Operand::Scalar { code, dwords: 1 }, Operand::Const(140)] if mul.read(Reg::S(*code)) == Some(result))
            { return None; }
        let (mut addr_reg, mut addr) = scalar_result(mul_i)?;
        let mut add_i = sole_use(mul_i, addr)?;
        if bb.insts[add_i].name() == "s_addk_i32" {
            let w = &bb.insts[add_i];
            let Fields::Sop { dst: Some(Operand::Scalar { code, dwords: 1 }), src, imm: Some(48) } = &w.fields else { return None; };
            if *code != addr_reg || !src.is_empty() || w.read(Reg::S(*code)) != Some(addr) { return None; }
            (addr_reg, addr) = scalar_result(add_i)?;
            add_i = sole_use(add_i, addr)?;
        } else if matches!(&bb.insts[add_i].fields, Fields::Sop { src, .. }
            if matches!(src.as_slice(), [Operand::Scalar { code, dwords: 1 }, Operand::Const(176)]
                if *code == addr_reg)) {
            // The original 0xc78 instruction encodes +48 as SOP2 with
            // inline constant 176 (128 + 48), not as SOPK addk. Require
            // the exact reaching address SSA and audit the new SCC too.
            let w = &bb.insts[add_i];
            if w.name() != "s_add_i32" || w.read(Reg::S(addr_reg)) != Some(addr) { return None; }
            (addr_reg, addr) = scalar_result(add_i)?;
            add_i = sole_use(add_i, addr)?;
        }
        let add = &bb.insts[add_i];
        let Fields::Sop { src, .. } = &add.fields else { return None; };
        let [Operand::Scalar { code: a @ 0..=105, dwords: 1 },
            Operand::Scalar { code: c @ 0..=105, dwords: 1 }] = src.as_slice() else { return None; };
        if add.name() != "s_add_i32" { return None; }
        let (lhs, rhs) = (add.read(Reg::S(*a))?, add.read(Reg::S(*c))?);
        let record = match (lhs == addr, rhs == addr) {
            (true, false) if *a == addr_reg => rhs,
            (false, true) if *c == addr_reg => lhs,
            _ => return None,
        };
        if !self.scalar_value_proven(record, &mut HashSet::new(), 0) { return None; }
        // The other summand is the uniform per-record 96-byte stride.
        let rm = bb.insts[..add_i].iter().find(|w| w.writes.iter().any(|&(_, v)| v == record))?;
        let Fields::Sop { src, .. } = &rm.fields else { return None; };
        if rm.name() != "s_mul_i32" || !matches!(src.as_slice(),
            [Operand::Scalar { code: 0..=105, dwords: 1 }, Operand::Literal(96)]) { return None; }
        let (addr_reg, addr) = scalar_result(add_i)?;
        let load_i = sole_use(add_i, addr)?;
        let load = &bb.insts[load_i];
        let Fields::Smem { data: Some(Operand::Scalar { code: out @ 0..=102, dwords: 4 }),
            offset: 0, soffset: Operand::Scalar { code, dwords: 1 }, .. } = &load.fields else { return None; };
        if load.name() != "s_buffer_load_dwordx4" || *code != addr_reg
            || load.read(Reg::S(*code)) != Some(addr) || load.writes.len() != 4 { return None; }
        let used = self.static_readonly_scalar_descriptor((b, load_i), load)?;
        if used.kind != ResourceType::Srv || used.stride != 96 { return None; }
        // Every consumed coefficient must go immediately into a same-mask
        // VGPR move, never a scalar phi, handle, mask, or later scalar read.
        let mut moves = 0;
        let mut last = load_i;
        for r in *out..*out + 4 {
            let val = load.write(Reg::S(r))?;
            let sites = uses(load_i, val)?;
            if r == *out + 3 {
                if !sites.is_empty() { return None; }
                continue;
            }
            let [j] = sites.as_slice() else { return None; };
            let w = &bb.insts[*j];
            let Fields::Valu(v) = &w.fields else { return None; };
            if w.name() != "v_mov_b32" || !plain(v) || !same_exec(w)
                || !matches!(v.src.as_slice(), [Operand::Scalar { code, dwords: 1 }] if *code == r)
                || w.read(Reg::S(r)) != Some(val) || w.writes.len() != 1
                || !matches!(v.dst, Some(Operand::Vector { reg, dwords: 1 }) if w.write(Reg::V(reg)).is_some())
                { return None; }
            moves += 1;
            last = last.max(*j);
        }
        if moves != 3 || bb.insts[cv_i..=last].iter().any(|w|
            w.writes.iter().any(|&(r, _)| r == Reg::S(EXEC) || r == Reg::S(EXEC + 1))) { return None; }
        Some(())
    }

    /// Only a static table SRV/CBV whose descriptor is the exact four
    /// scalar SSA outputs of its tracked table load. No dynamic heap,
    /// per-lane, built, selected or waterfall descriptor is admitted.
    fn static_readonly_scalar_descriptor(&self, site: Site, inst: &Inst) -> Option<&'a UsedResource> {
        let Fields::Smem { base: Operand::Scalar { code, dwords: 4 }, cache, .. } = &inst.fields else { return None; };
        if *code > 102 || *cache != crate::gfx10::Cache::default()
            || self.x.res.waterfall.loads.contains(&site) { return None; }
        let src = *self.descriptors.get(&(site, false))?;
        let Source::Loaded { load, first: 0 } = src else { return None; };
        let address = self.scalar.get(load)?;
        if address.dynamic.is_some() || address.offset % 4 != 0 { return None; }
        let root = self.x.native.layout.get(address.param)?;
        if root.root != RootType::Table { return None; }
        let used = self.used_kind(src)?;
        if !matches!(used.kind, ResourceType::Srv | ResourceType::Cbv)
            || used.dwords != 4 || used.table_offset_dwords != address.offset / 4 { return None; }
        let desc = self.x.p.blocks.get(load.0)?.insts.get(load.1)?;
        let Fields::Smem { data: Some(Operand::Scalar { code: dc @ 0..=102, dwords: 4 }), .. } = &desc.fields else { return None; };
        if desc.name() != "s_load_dwordx4" { return None; }
        for k in 0..4 {
            let value = inst.read(Reg::S(*code + k))?;
            if desc.write(Reg::S(*dc + k)) != Some(value)
                || !self.scalar_value_proven(value, &mut HashSet::new(), 0) { return None; }
        }
        Some(used)
    }

    /// Complete scalar-use audit; all reads must be local and unguarded.
    /// No scalar value certified by the conversion can escape as a phi,
    /// even via a dead-looking SCC result of the address arithmetic.
    fn partial_data_local_uses(&self, block: usize, after: usize, value: Value) -> Option<Vec<usize>> {
        let mut uses = Vec::new();
        for (b, bb) in self.x.p.blocks.iter().enumerate() {
            if bb.phis.iter().any(|p| p.args.contains(&value)) { return None; }
            let masks = self.x.st.guards[b].iter().flatten().copied().chain(
                match self.x.st.exits[b] { LaneExit::Branch { when, .. } => Some(when), _ => None });
            if masks.into_iter().any(|m| self.x.st.bools.support(m).into_iter().any(|a|
                matches!(a, Atom::Uniform(v) | Atom::Lane(v) | Atom::Opaque(v) | Atom::Phi(v) | Atom::Quad(v) if v == value))) { return None; }
            for (i, w) in bb.insts.iter().enumerate() {
                if w.reads.iter().any(|&(_, v)| v == value) {
                    if b != block || i <= after || self.x.st.guards[b][i].is_some() { return None; }
                    uses.push(i);
                }
            }
        }
        Some(uses)
    }

    fn uniform_scalar_select_proven(
        &self, read: &Inst, visiting: &mut HashSet<Value>, depth: usize,
    ) -> bool {
        if depth >= 96 { return false }
        let Some(Fields::Valu(rv)) = Some(&read.fields) else { return false };
        let Some(Operand::Vector { reg, .. }) = rv.src.first() else { return false };
        let Some(value) = read.read(Reg::V(*reg)) else { return false };
        let Some((block, i, writer)) = self.x.p.blocks.iter().enumerate()
            .flat_map(|(b, block)| block.insts.iter().enumerate().map(move |(i, w)| (b, i, w)))
            .find(|(_, _, w)| w.writes.iter().any(|&(_, v)| v == value)) else { return false };
        if writer.name() != "v_cndmask_b32"
            || self.x.st.guards[block][i].is_some()
            || read.read(Reg::S(EXEC)) != writer.read(Reg::S(EXEC))
            || read.read(Reg::S(EXEC + 1)) != writer.read(Reg::S(EXEC + 1))
            || !self.full_exec_at(read) {
            return false;
        }
        let Fields::Valu(wv) = &writer.fields else { return false };
        if wv.src.len() != 3 || wv.abs != 0 || wv.neg != 0 || wv.opsel != 0
            || wv.sdwa.is_some() || wv.dpp.is_some() { return false; }
        if !wv.src[..2].iter().all(|op| match *op {
            Operand::Const(_) | Operand::Literal(_) => true,
            Operand::Scalar { code, dwords: 1 } => writer.read(Reg::S(code))
                .is_some_and(|v| self.scalar_value_proven(v, visiting, depth + 1)),
            _ => false,
        }) {
            return false;
        }
        let Some(&Operand::Scalar { code, dwords: 2 }) = wv.src.get(2) else { return false };
        let Some(lo) = writer.read(Reg::S(code)) else { return false };
        let Some(hi) = writer.read(Reg::S(code + 1)) else { return false };
        let Some(mask) = self.x.st.mask_bits(lo, Some(hi)) else { return false };
        self.x.st.bools.support(mask).into_iter().all(|a| match a {
            Atom::Uniform(v) => self.scalar_value_proven(v, visiting, depth + 1),
            _ => false,
        })
    }

    /// Require every original console lane to be active. Merely comparing
    /// writer/read EXEC SSA is insufficient: inactive VGPR lanes may retain
    /// earlier divergent values that a scalar read could observe later.
    fn full_exec_at(&self, inst: &Inst) -> bool {
        let (Some(lo), Some(hi)) = (inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1))) else {
            return false;
        };
        self.x.st.mask_bits(lo, Some(hi)) == Some(crate::structure::Bool::TRUE)
    }

    /// A buffer fetch at a scalar-computed index is identical for every
    /// executing lane of a console wave: the GPU has one scalar descriptor
    /// and all lanes fetch the same element. Only apply that identity when
    /// the shader contains no buffer/image stores or atomics, the index is
    /// demonstrably a scalar-to-vector move, and both operations execute
    /// under exactly the same EXEC as the first-lane read. Otherwise leave
    /// the actual wave operation and the host's wave64 guard intact.
    fn uniform_buffer_fetch_at_firstlane(&self, read: &Inst) -> bool {
        self.uniform_buffer_fetch_proven(read, &mut HashSet::new(), 0)
    }

    fn uniform_buffer_fetch_proven(
        &self, read: &Inst, visiting: &mut HashSet<Value>, depth: usize,
    ) -> bool {
        if depth >= 96 { return false }
        let Fields::Valu(rv) = &read.fields else { return false };
        let Some(Operand::Vector { reg, dwords: 1 }) = rv.src.first() else { return false };
        let Some(value) = read.read(Reg::V(*reg)) else { return false };
        if self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
            let name = inst.name();
            name.contains("_store") || name.contains("atomic")
                || name.starts_with("ds_write") || name.starts_with("flat_store")
        }) { return false; }
        let Some((block, i, fetched)) = self.x.p.blocks.iter().enumerate()
            .flat_map(|(b, block)| block.insts.iter().enumerate().map(move |(i, w)| (b, i, w)))
            .find(|(_, _, w)| w.writes.iter().any(|&(_, v)| v == value)) else { return false };
        let partial = !self.full_exec_at(read);
        if (!partial && fetched.name() != "buffer_load_format_x")
            || (partial && fetched.name() != "buffer_load_format_xy")
            || self.x.st.guards[block][i].is_some()
            || read.read(Reg::S(EXEC)) != fetched.read(Reg::S(EXEC))
            || read.read(Reg::S(EXEC + 1)) != fetched.read(Reg::S(EXEC + 1))
        {
            return false;
        }
        let Fields::Buffer {
            addr: Operand::Vector { reg: address_reg, dwords: 1 },
            rsrc: Operand::Scalar { code: descriptor, dwords: 4 },
            data, soffset, offset,
            idxen: true, offen: false, lds: false, tfe: false, ..
        } = &fetched.fields else { return false };
        let Some(index) = fetched.read(Reg::V(*address_reg)) else { return false };
        let Some((source_block, source_i, source)) = self.x.p.blocks.iter().enumerate()
            .flat_map(|(b, block)| block.insts.iter().enumerate().map(move |(i, w)| (b, i, w)))
            .find(|(_, _, w)| w.writes.iter().any(|&(_, v)| v == index)) else { return false };
        let Fields::Valu(mov) = &source.fields else { return false };
        if partial {
            // Only a local, read-only typed fetch in the exact active region.
            // Its scalar result must never escape into lanes that skipped it.
            let Some(read_i) = self.x.p.blocks[block].insts.iter().position(|inst| std::ptr::eq(inst, read)) else { return false; };
            let Some(Operand::Vector { reg: first, dwords: 2 }) = *data else { return false; };
            if !self.x.native.wave64 || self.s.mesh.is_none()
                || source_block != block || !(source_i < i && i < read_i)
                || self.x.st.guards[block][read_i].is_some()
                || !matches!(mov.src.as_slice(), [Operand::Scalar { code: 0..=105, dwords: 1 }])
                || !matches!(mov.dst, Some(Operand::Vector { reg, dwords: 1 }) if reg == *address_reg)
                || !matches!(rv.dst, Some(Operand::Scalar { code: 0..=105, dwords: 1 }))
                || rv.abs != 0 || rv.neg != 0 || rv.opsel != 0 || rv.clamp || rv.omod != 0
                || rv.sdwa.is_some() || rv.dpp.is_some() || rv.src.len() != 1
                || mov.clamp || mov.omod != 0 || mov.neg_hi != 0 || mov.opsel_hi != 0
                || first.checked_add(1) != Some(*reg)
                || fetched.write(Reg::V(*reg)) != Some(value)
                || *offset != 0 || *soffset != Operand::Const(128) || *descriptor > 102
                || self.sections.get(&block).is_some_and(|ss| ss.iter().any(|(&start, sec)| start <= read_i && sec.end > source_i))
                || !matches!((read.read(Reg::S(EXEC)), read.read(Reg::S(EXEC + 1))),
                    (Some(lo), Some(hi)) if self.x.st.mask_bits(lo, Some(hi)).is_some_and(|m| m != Bool::FALSE))
                || !(*descriptor..*descriptor + 4).all(|r| fetched.read(Reg::S(r))
                    .is_some_and(|v| self.scalar_value_proven_impl(v, visiting, depth + 1, true)))
                || !read.writes.first().is_some_and(|&(r, v)| matches!(r, Reg::S(0..=105)) && !self.x.res.waterfall.reads.contains(&v)
                    && self.partial_fetch_local_uses(block, read_i, read, v, 0)) {
                return false;
            }
        }
        source.name() == "v_mov_b32"
            && self.x.st.guards[source_block][source_i].is_none()
            && source.read(Reg::S(EXEC)) == fetched.read(Reg::S(EXEC))
            && source.read(Reg::S(EXEC + 1)) == fetched.read(Reg::S(EXEC + 1))
            && mov.neg == 0 && mov.abs == 0 && mov.opsel == 0
            && mov.sdwa.is_none() && mov.dpp.is_none()
            && match mov.src.first() {
                Some(Operand::Const(_)) => true,
                Some(&Operand::Scalar { code, dwords: 1 }) => source.read(Reg::S(code))
                    .is_some_and(|v| self.scalar_value_proven_impl(v, visiting, depth + 1, partial)),
                _ => false,
            }
    }

    /// Partial EXEC first-lane results are not global SGPR proofs. Accept
    /// only a local scalar add chain ending in ordinary per-lane adds with
    /// the same EXEC; no phi, branch, scalar escape or lane operation.
    fn partial_fetch_local_uses(&self, block: usize, after: usize, read: &Inst, value: Value, depth: usize) -> bool {
        if depth >= 16 { return false; }
        for (b, bb) in self.x.p.blocks.iter().enumerate() {
            if bb.phis.iter().any(|p| p.args.contains(&value)) { return false; }
            for (i, inst) in bb.insts.iter().enumerate() {
                if !inst.reads.iter().any(|&(_, v)| v == value) { continue; }
                if b != block || i <= after || self.x.st.guards[b][i].is_some() { return false; }
                match inst.name() {
                    "s_add_i32" if inst.writes.iter().all(|&(r, v)| matches!(r, Reg::S(0..=105) | Reg::Scc)
                        && self.partial_fetch_local_uses(block, i, read, v, depth + 1)) => {},
                    "v_add_nc_u32" if inst.read(Reg::S(EXEC)) == read.read(Reg::S(EXEC))
                        && inst.read(Reg::S(EXEC + 1)) == read.read(Reg::S(EXEC + 1))
                        && matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_none())
                        && inst.writes.iter().all(|&(r, _)| matches!(r, Reg::V(_))) => {},
                    _ => return false,
                }
            }
        }
        true
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if !self.logical_lane_sites.is_empty() {
            return self.mesh_thread();
        }
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // From 0xE000: an FFT order; from 0xC000: a rotation. Below, bit
            // 15 set: each lane of a quad picks a lane of it; else lanes of
            // each group of 32 combine their index with and/or/xor masks.
            let (fft, rotate) = (offset >= 0xE000, (0xC000..0xE000).contains(&offset));
            let quad = !(fft || rotate)
                && (offset & 0x8000 != 0
                    || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4);
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            let logical = self.logical_lane_sites.get(&(b, i)).copied();
            if !mesh_quad && logical.is_none() {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if fft || rotate {
                // Both stay within the lane's group of 32.
                let (low, high) = (k(self, 0x1F), k(self, !0x1F));
                let mask = offset & 0x1F;
                let i = self.s.m.bin(c, Bin::And, lane, low);
                let j = if fft {
                    // The 5 index bits reversed, shifted down by the number
                    // of mask bits; the mask's bits stay the lane's own.
                    let (top, m, i32t) = (k(self, 27), k(self, mask), self.s.t.i32);
                    let down = k(self, mask.count_ones());
                    let j = self.s.op(c, "unary.i32", Attr::ReadNone, i32t, 30, &[i]);
                    let j = self.s.m.bin(c, Bin::LShr, j, top);
                    let j = self.s.m.bin(c, Bin::LShr, j, down);
                    let own = self.s.m.bin(c, Bin::And, i, m);
                    self.s.m.bin(c, Bin::Or, j, own)
                } else {
                    // Left by bits 5-9 lanes (bit 10: right); the mask's
                    // bits stay the lane's own. This is the ISA's
                    // pseudocode, which RDNA2 hardware follows; its
                    // examples for mask 1 list other lanes.
                    let n = offset >> 5 & 0x1F;
                    let n = if offset & 0x400 != 0 {
                        n.wrapping_neg()
                    } else {
                        n
                    };
                    let (n, m, rest) = (k(self, n), k(self, mask), k(self, !mask & 0x1F));
                    let j = self.s.m.bin(c, Bin::Add, i, n);
                    let j = self.s.m.bin(c, Bin::And, j, rest);
                    let own = self.s.m.bin(c, Bin::And, i, m);
                    self.s.m.bin(c, Bin::Or, j, own)
                };
                let group = self.s.m.bin(c, Bin::And, lane, high);
                self.s.m.bin(c, Bin::Or, group, j)
            } else if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if let Some((base, publish)) = logical {
                self.logical_lane_read(base, publish, x, src)
            } else if mesh_quad {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        if name == "ds_write_b16" {
            // Native LDS halfword stores preserve the other 16 bits of the
            // dword. Two independent atomic bit operations prevent a lane
            // writing the other half from being lost on the 32-lane host.
            let p = dword(self, offset);
            let add = self.s.m.i32(offset);
            let byte = self.s.m.bin(c, Bin::Add, a, add);
            let three = self.s.m.i32(3);
            let within = self.s.m.bin(c, Bin::And, byte, three);
            let eight = self.s.m.i32(8);
            let shift = self.s.m.bin(c, Bin::Shl, within, eight);
            let val = self.vgprs(inst, data.first().ok_or("half LDS store data")?, 1)?[0];
            let mask = self.s.m.i32(0xffff);
            let x = self.s.m.bin(c, Bin::And, val, mask);
            let x = self.s.m.bin(c, Bin::Shl, x, shift);
            let clear = self.s.m.bin(c, Bin::Shl, mask, shift);
            let ones = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, clear, ones);
            let at = self.begin_guarded(guard);
            self.s.m.atomic(self.cur, super::bitcode::Rmw::And, p, keep);
            self.s.m.atomic(self.cur, super::bitcode::Rmw::Or, p, x);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    // ---- instructions ----

    fn allocation_plan_proven(&self, plan: LdsAllocation) -> bool {
        self.allocation_memory_order(plan) && plan.reciprocal.is_none_or(|(b,i)| {
            let read = &self.x.p.blocks[b].insts[i];
            self.x.st.guards[b][i].is_none()
                && !read.writes.iter().any(|&(_,v)| self.x.res.waterfall.reads.contains(&v))
                && self.uniform_arithmetic_firstlane(read, &mut HashSet::new(), 0)
        })
    }

    /// Identify an exact static table descriptor, not merely the resource
    /// range containing an arbitrary dynamic index.
    fn allocation_resource(&self, site: Site, inst: &Inst) -> Option<&'a UsedResource> {
        let operand = match &inst.fields {
            Fields::Buffer { rsrc, .. } => rsrc,
            Fields::Smem { base, .. } => base,
            _ => return None,
        };
        let Operand::Scalar { code, dwords: 4 } = *operand else { return None; };
        if code > 102 { return None; }
        let src = *self.descriptors.get(&(site,false))?;
        let Source::Loaded { load, first: 0 } = src else { return None; };
        let a = self.scalar.get(load)?;
        let root = self.x.native.layout.get(a.param)?;
        if a.dynamic.is_some() || a.offset % 4 != 0 || root.root != RootType::Table
            || self.x.res.waterfall.loads.contains(load) { return None; }
        let used = self.used_kind(src)?;
        if used.dwords != 4 || used.table_offset_dwords != a.offset / 4 { return None; }
        let desc = self.x.p.blocks.get(load.0)?.insts.get(load.1)?;
        let Fields::Smem { data: Some(Operand::Scalar { code: dc, dwords: 4 }), .. } = desc.fields else { return None; };
        if dc > 102 || desc.name() != "s_load_dwordx4" { return None; }
        for k in 0..4 {
            let v = inst.read(Reg::S(code+k))?;
            if desc.write(Reg::S(dc+k)) != Some(v)
                || !self.scalar_value_proven(v, &mut HashSet::new(), 0) { return None; }
        }
        Some(used)
    }

    /// Within each logical output resource the original writes touch
    /// nonoverlapping fields of a structured record, with the same SSA
    /// record index for every field. Thus the independently scheduled host
    /// lanes need no further cross-lane ordering between those writes.
    /// The raw-buffer test variant proves its disjoint byte addresses.
    /// All data loads after the rendezvous are from a static read-only view.
    fn allocation_memory_order(&self, plan: LdsAllocation) -> bool {
        let p = self.x.p;
        let mut todo = vec![plan.gate];
        let mut seen = HashSet::new();
        let mut counter = None;
        let mut outputs: HashMap<(u8,u32), (Value,Vec<(u32,u32)>)> = HashMap::new();
        let mut raw_counter = false;
        let mut raw_store = false;
        let def = |v: Value| p.blocks.iter().enumerate().flat_map(|(b,bb)|
            bb.insts.iter().enumerate().map(move |(i,w)| ((b,i),w)))
            .find(|(_,i)| i.writes.iter().any(|&(_,w)| w == v));
        let key = |u: &UsedResource| (u.user_data, u.table_offset_dwords);
        let exec = |w: &Inst| (w.read(Reg::S(EXEC)),w.read(Reg::S(EXEC+1)));
        let plain = |f: &crate::gfx10::Valu| f.neg == 0 && f.abs == 0 && f.opsel == 0
            && f.neg_hi == 0 && f.opsel_hi == 0 && !f.clamp && f.omod == 0
            && f.sdwa.is_none() && f.dpp.is_none() && f.carry.is_none();
        let mov_const = |v: Value, k: u8, site: Site, use_inst: &Inst| def(v).is_some_and(|((b,i),w)|
            b == site.0 && i < site.1 && self.x.st.guards[b][i].is_none() && exec(w) == exec(use_inst)
                && w.name() == "v_mov_b32"
                && matches!(&w.fields, Fields::Valu(f) if plain(f) && f.src.as_slice() == [Operand::Const(k)]));
        let raw_addr = |v: Value, site: Site, use_inst: &Inst| -> Option<()> {
            let ((sb,si),sh) = def(v)?;
            let Fields::Valu(s) = &sh.fields else { return None; };
            let [Operand::Const(130), Operand::Vector { reg, dwords: 1 }] = s.src.as_slice() else { return None; };
            if sh.name() != "v_lshlrev_b32" || sb != site.0 || si >= site.1 || exec(sh) != exec(use_inst)
                || self.x.st.guards[sb][si].is_some() { return None; }
            let ((ab,ai),add) = def(sh.read(Reg::V(*reg))?)?;
            let Fields::Valu(a) = &add.fields else { return None; };
            let [Operand::Const(129), Operand::Vector { reg, dwords: 1 }] = a.src.as_slice() else { return None; };
            if add.name() != "v_add_nc_u32" || ab != sb || ai >= si || exec(add) != exec(sh)
                || self.x.st.guards[ab][ai].is_some() { return None; }
            let ((mb,mi),mov) = def(add.read(Reg::V(*reg))?)?;
            let Fields::Valu(m) = &mov.fields else { return None; };
            let [Operand::Vector { reg, dwords: 1 }] = m.src.as_slice() else { return None; };
            if mov.name() != "v_mov_b32" || mb != 0 || !self.full_exec_at(mov)
                || self.x.st.guards[mb][mi].is_some()
                || !self.x.io.inputs.contains(&(mov.read(Reg::V(*reg))?,Input::ThreadIdInGroup(0))) { return None; }
            for f in [s,a,m] {
                if f.neg != 0 || f.abs != 0 || f.opsel != 0 || f.neg_hi != 0 || f.opsel_hi != 0
                    || f.clamp || f.omod != 0 || f.sdwa.is_some() || f.dpp.is_some() || f.carry.is_some() { return None; }
            }
            Some(())
        };
        while let Some(b) = todo.pop() {
            if !seen.insert(b) { continue; }
            todo.extend(p.blocks[b].exit.successors());
            for (i, w) in p.blocks[b].insts.iter().enumerate() {
                let n = w.name();
                match &w.fields {
                    Fields::Smem { .. } if n.starts_with("s_buffer_load_") => {
                        if self.static_readonly_scalar_descriptor((b,i), w).is_none()
                            || !w.reads.iter().all(|&(_,v)| self.scalar_value_proven(v,&mut HashSet::new(),0)) { return false; }
                    }
                    Fields::Smem { .. } => {
                        if n != "s_load_dwordx4" || !self.scalar.get(&(b,i)).is_some_and(|a|
                            a.dynamic.is_none() && self.x.native.layout.get(a.param).is_some_and(|r| r.root == RootType::Table)) { return false; }
                    }
                    Fields::Buffer { addr: Operand::Vector { reg, dwords: 1 }, data: Some(Operand::Vector { dwords, .. }),
                        soffset: Operand::Const(128), offset, idxen, offen, format: None, lds: false, tfe: false, .. } => {
                        let Some(u) = self.allocation_resource((b,i),w) else { return false; };
                        let Some(index) = w.read(Reg::V(*reg)) else { return false; };
                        if u.kind != ResourceType::Uav || idxen == offen { return false; }
                        if n == "buffer_atomic_add" {
                            if counter.replace(key(u)).is_some() { return false; }
                            raw_counter = *offen;
                            if raw_counter && (*offset != 0 || !mov_const(index,128,(b,i),w)) { return false; }
                            if !raw_counter && u.stride < 4 { return false; }
                        } else if n.starts_with("buffer_store_dword") {
                            if *offen {
                                if raw_store || *dwords != 1 || *offset != 0 || raw_addr(index,(b,i),w).is_none() { return false; }
                                raw_store = true;
                                if !raw_counter || counter != Some(key(u)) { return false; }
                            } else {
                                let begin = u32::from(*offset);
                                let end = begin + 4*u32::from(*dwords);
                                if u.stride == 0 || begin % 4 != 0 || end > u.stride { return false; }
                                let (old,ranges) = outputs.entry(key(u)).or_insert_with(|| (index,Vec::new()));
                                if *old != index || ranges.iter().any(|&(a,b)| begin < b && a < end) { return false; }
                                ranges.push((begin,end));
                            }
                        } else { return false; }
                    }
                    Fields::Buffer { .. } | Fields::Image { .. } | Fields::Flat { .. } => return false,
                    _ => {}
                }
            }
        }
        if raw_counter {
            raw_store && outputs.is_empty()
        } else {
            !raw_store && counter.is_some_and(|c| !outputs.contains_key(&c)) && !outputs.is_empty()
        }
    }

    fn allocation_sync(&mut self, mode: u32) {
        let mode = self.s.m.i32(mode);
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[mode]);
    }

    fn allocation_abort_ptr(&mut self, slot: u32) -> V {
        let lds = self.s.lds();
        let idx = self.s.m.i32(slot);
        self.s.m.element(self.cur, lds, idx)
    }

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        if let Some(plan) = self.lds_allocation {
            if plan.init == (b, i) {
                let tid = self.mesh_thread();
                let zero = self.s.m.i32(0);
                let leader = self.s.m.cmp(self.cur, pred::EQ, tid, zero);
                let at = self.begin_guarded(Some(leader));
                let ptr = self.allocation_abort_ptr(plan.abort_slot);
                self.s.m.store(self.cur, ptr, zero);
                self.end_guarded(at, &[]);
                self.allocation_sync(9);
            }
        }
        if let Some((site, mode)) = self.implicit_lds_barrier && site == (b, i) {
            let mode = self.s.m.i32(mode); // group sync + TGSM, UAV fence for image filter
            self.s.op(self.cur, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[mode]);
        }
        if self.matrix_lds_barriers.is_some_and(|(publish,retire)|publish==(b,i)||retire==(b,i)) {
            self.allocation_sync(11);
        }
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                // Omit SYNC_THREAD_GROUP (1) when control-dependent on a
                // per-lane branch, where a 32-lane host deadlocks if one
                // warp skips the barrier while another waits on it.
                let sync = if self.fence_only_barrier.contains(&b) {
                    0
                } else {
                    1
                };
                let mode = sync | 8 | if self.writes_uavs { 2 } else { 0 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        if name == "s_movrels_b32" {
            let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
            let Some(&Operand::Scalar { code: sreg, .. }) = src.first() else {
                return Err("s_movrels_b32 source operand");
            };
            if sreg >= 106 {
                return Err("s_movrels_b32 base out of SGPR range");
            }
            let lo = lo.ok_or("s_movrels_b32 destination")?;
            let r = if let Some(m0_val) = self.s.m.as_i32(m0) {
                if m0_val >= 0 && (m0_val as u64) <= (105 - sreg) as u64 {
                    let target = sreg + m0_val as u8;
                    match inst.read(Reg::S(target)).map(|v| self.i32_of(v)) {
                        Some(Ok(x)) => x,
                        _ => self.s.m.i32(0),
                    }
                } else {
                    self.s.m.i32(0)
                }
            } else {
                let mut r = self.s.m.i32(0);
                for k in 0..=105u8.saturating_sub(sreg) {
                    let x = match inst.read(Reg::S(sreg + k)).map(|x| self.i32_of(x)) {
                        Some(Ok(x)) => x,
                        _ => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    r = self.s.m.select(c, is, x, r);
                }
                r
            };
            self.set(Some(lo), Val::I32(r));
            return Ok(());
        }
        if name == "s_movreld_b32" {
            let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
            let dreg = dst_code.ok_or("s_movreld_b32 destination operand")?;
            if dreg >= 106 {
                return Err("s_movreld_b32 base out of SGPR range");
            }
            let value = src32(self, 0)?;
            if let Some(m0_val) = self.s.m.as_i32(m0) {
                let target = if m0_val >= 0 && (m0_val as u64) <= (105 - dreg) as u64 {
                    Some(dreg + m0_val as u8)
                } else {
                    None
                };
                for k in 0..=105u8.saturating_sub(dreg) {
                    let reg = dreg + k;
                    let Some(_) = inst.write(Reg::S(reg)) else {
                        break;
                    };
                    let x = if Some(reg) == target {
                        value
                    } else {
                        match inst.read(Reg::S(reg)).map(|o| self.i32_of(o)) {
                            Some(Ok(o)) => o,
                            _ => self.s.m.i32(0),
                        }
                    };
                    self.write_placed(inst, Reg::S(reg), x, None);
                }
            } else {
                for k in 0..=105u8.saturating_sub(dreg) {
                    let reg = dreg + k;
                    let Some(_) = inst.write(Reg::S(reg)) else {
                        break;
                    };
                    let old = match inst.read(Reg::S(reg)).map(|o| self.i32_of(o)) {
                        Some(Ok(o)) => o,
                        _ => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::S(reg), x, None);
                }
            }
            return Ok(());
        }
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let size = self.root_view_size(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if let Some(&entry) = self.static_cbv_sites.get(&site) {
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.static_cbv_dword(entry, a);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let h = self.handle(site, src, false, (11, 0))?;
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (a, ok) = match size {
                    Some(size) => {
                        let (ok, a) = self.in_root_view(size, a, 4);
                        (a, Some(ok))
                    }
                    None => (a, None),
                };
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                let x = self.zero_unless(ok, x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    match size {
                        Some(size) => {
                            let (ok, a) = self.in_root_view(size, a, 4);
                            let x = self.cbuffer_dword(h, a);
                            self.zero_unless(Some(ok), x)
                        }
                        None => self.cbuffer_dword(h, a),
                    }
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    /// Bindless image descriptors read only to calculate their dimensions have
    /// no direct image instruction to identify the view. Require the GFX10
    /// width, height, and base-mip extraction pattern, and evidence of 2D
    /// sampling from the same heap, before reconstructing their dimensions.
    fn bindless_2d_size_load(&self, inst: &Inst, site: Site, code: u8) -> bool {
        let Some(a) = self.scalar.get(&site) else {
            return false;
        };
        if !matches!(self.x.native.layout.get(a.param), Some(p) if p.root == RootType::ResourceHeap)
            || a.dynamic.is_none()
            || a.offset != 0
        {
            return false;
        }
        let [Some(d1), Some(d2), Some(d3)] = [
            inst.write(Reg::S(code + 1)),
            inst.write(Reg::S(code + 2)),
            inst.write(Reg::S(code + 3)),
        ] else {
            return false;
        };
        let has = |value, name, second| {
            self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                i.name() == name
                    && i.reads.iter().any(|&(_, v)| v == value)
                    && matches!(&i.fields, Fields::Sop { src, .. } if src.get(1) == Some(&second))
            })
        };
        // d1[31:30], d2[13:0] => width - 1; d2[29:14] => height - 1;
        // d3[15:12] => base mip. No arbitrary descriptor words are invented.
        if !has(d1, "s_lshr_b32", Operand::Const(158))
            || !has(d2, "s_lshl_b32", Operand::Const(130))
            || !has(d2, "s_bfe_u32", Operand::Literal(0x0010_000e))
            || !has(d3, "s_bfe_u32", Operand::Literal(0x0004_000c))
        {
            return false;
        }
        self.x.res.descriptors.iter().any(|d| {
            let use_inst = &self.x.p.blocks[d.site.0].insts[d.site.1];
            d.role == Role::Resource
                && use_inst.name().starts_with("image_sample")
                && matches!(&use_inst.fields, Fields::Image { dim: 1, .. })
                && d.source.leaves().iter().any(|source| {
                    matches!(source, Source::Loaded { load, .. } if self.scalar.get(load).is_some_and(|other| other.param == a.param))
                })
        })
    }

    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        if self.atomic_descriptor_loads.contains(&site) {
            return Ok(());
        }
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind.
        let use_at = self
            .x
            .res
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
            .find_map(|d| {
                d.source.leaves().into_iter().find_map(|l| match *l {
                    Source::Loaded { load, first } if load == site => Some((d.site, first)),
                    _ => None,
                })
            });
        let mut scalar_record_stride = None;
        let (src, uav, props, image) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        (10 | u, 0x409)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4)
                    {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_load_dword") => {
                    // Scalar buffer loads use the raw alias of the original
                    // structured SRV. GFX10 descriptor dword 2 is a record
                    // count, whereas DXIL raw GetDimensions returns bytes.
                    // Require the exact static table entry and a supported
                    // original stride; don't reconstruct arbitrary descriptor
                    // words or infer a count from another resource.
                    let Some(used) = used.filter(|u| u.kind == ResourceType::Srv
                        && u.root == RootType::Table && u.dwords == 4
                        && u.stride >= 4 && u.stride.is_multiple_of(4)) else {
                        return Err("a scalar memory read as data");
                    };
                    if count != 4 || first != 0 || !self.scalar.get(&site)
                        .is_some_and(|a| a.dynamic.is_none() && a.offset % 4 == 0
                            && a.offset / 4 == used.table_offset_dwords) {
                        return Err("a scalar memory read as data");
                    }
                    scalar_record_stride = Some(used.stride);
                    ((11, 0), None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (src, uav, props, image)
        } else {
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            if count == 8 && self.bindless_2d_size_load(inst, site, code) {
                // This handle uses the original descriptor's *own* dynamic
                // heap index, not the different indices sampled afterwards.
                (src, false, (2, 0x409), Some(2))
            } else {
                let Some(used) = self.used_kind(&src) else {
                    return Err("a descriptor read only as data");
                };
                if used.kind == ResourceType::Cbv {
                    return Err("a constant buffer's descriptor read as data");
                }
                if !matches!(used.kind, ResourceType::Srv | ResourceType::Uav) {
                    return Err("a descriptor read only as data");
                }
                let uav = used.kind == ResourceType::Uav;
                let u = if uav { 0x1000 } else { 0 };
                let (props, image) = if used.dwords == 4 {
                    let stride = used.stride;
                    let props = if stride >= 4 && stride.is_multiple_of(4) {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    (props, None)
                } else {
                    ((2 | u, 0x409), Some(2))
                };
                (src, uav, props, image)
            }
        };
        let Source::Loaded { first, .. } = src else {
            unreachable!()
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => {
                let records = field(self, 0);
                let records = if let Some(stride) = scalar_record_stride {
                    let stride = self.s.m.i32(stride);
                    self.s.m.bin(c, Bin::UDiv, records, stride)
                } else {
                    records
                };
                vec![(2, records)]
            }
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// A raw integer load: preserve every CBV bit pattern, including NaNs.
    fn static_cbv_raw(&mut self, c: usize, h: V, a: V) -> V {
        let i = self.s.t.i32;
        let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i; 5]);
        let u = self.s.m.undef(i);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(c, "rawBufferLoad.i32", Attr::ReadOnly, ret, 139, &[h, a, u, mask, align]);
        self.s.m.extract(c, r, 0)
    }

    /// The packed scratch root SRV has no hardware descriptor bounds.
    /// Reproduce each original CBV's own bounds before the raw read, not
    /// the entire scratch range. Invalid reads go to its zero sentinel.
    /// The header is immutable for a recorded draw and valid even for a
    /// null CBV; all payload offsets are aligned and checked by the host.
    fn static_cbv_dword(&mut self, entry: usize, a: V) -> V {
        let h = self.s.root_srv_handle(super::STATIC_CBV_SCRATCH);
        let (base, size) = match self.static_cbv_headers.get(&entry).copied() {
            Some(v) => v,
            None => {
                let c = self.s.entry;
                let base_at = self.s.m.i32(8 * entry as u32);
                let size_at = self.s.m.i32(8 * entry as u32 + 4);
                let base = self.static_cbv_raw(c, h, base_at);
                let size = self.static_cbv_raw(c, h, size_at);
                self.static_cbv_headers.insert(entry, (base, size));
                (base, size)
            }
        };
        let c = self.cur;
        // cbufferLoadLegacy selects a row and a dword within that row;
        // match that byte-address rounding exactly.
        let aligned = self.s.m.i32(!3u32);
        let a = self.s.m.bin(c, Bin::And, a, aligned);
        let (ok, a) = self.in_root_view(size, a, 4);
        let byte = self.s.m.bin(c, Bin::Add, base, a);
        let sentinel = self.s.m.i32(8 * self.static_table_cbvs.len() as u32);
        let safe = self.s.m.select(c, ok, byte, sentinel);
        let value = self.static_cbv_raw(c, h, safe);
        self.zero_unless(Some(ok), value)
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    // Native raw UAV buffer reads and writes use the same
                    // root descriptor and the existing bounded offset path.
                    (RootType::Descriptor, Some(ResourceType::Uav), 11 | 0x100b) => {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 0x100b) => {
                        Err("a write through a root SRV")
                    }
                    _ => Err("a typed root view"),
                }
            }
            Source::Phi { block, args } => {
                // Each predecessor computes its heap index at its end; the
                // join picks one with a phi.
                let _ = site;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

    /// The source heap descriptor's runtime bytes per buffer element.
    /// This is read from the host's table at the *original* heap index,
    /// never guessed from another resource in the shader.
    fn slot_stride(&mut self, src: &Source) -> Result<Option<V>, Why> {
        let Source::Loaded { load, first } = src else { return Ok(None) };
        if !self.scalar.get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap)
        {
            return Ok(None);
        }
        let c = self.cur;
        let slot = self.heap_index(*load, *first, false)?;
        // Dword 0 is the host heap's actual title slot count. Dwords
        // 1..count are real strides and count+1 is a permanently zeroed
        // sentinel. Thus an invalid index never reads past a small heap's
        // allocation or inherits an unrelated slot's element stride.
        let i32t = self.s.t.i32;
        let two = self.s.m.i32(2);
        let handle = self.s.root_srv_handle(super::shader::SLOT_STRIDES);
        let undef = self.s.m.undef(i32t);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let zero = self.s.m.i32(0);
        let count_result = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret,
            139, &[handle, zero, undef, mask, align]);
        let count_field = self.s.m.extract(c, count_result, 0);
        let count = self.as_int(count_field);
        let max = self.s.m.i32(super::shader::SLOT_STRIDES_COUNT);
        let allowed = self.s.m.cmp(c, super::bitcode::pred::IULT, slot, count);
        let within_max = self.s.m.cmp(c, super::bitcode::pred::IULT, slot, max);
        let valid = self.s.m.bin(c, Bin::And, allowed, within_max);
        let selected = self.s.m.select(c, valid, slot, count);
        let one = self.s.m.i32(1);
        let offset = self.s.m.bin(c, Bin::Add, selected, one);
        let byte = self.s.m.bin(c, Bin::Shl, offset, two);
        let result = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret,
            139, &[handle, byte, undef, mask, align]);
        let value = self.s.m.extract(c, result, 0);
        Ok(Some(self.as_int(value)))
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard, b, i);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" | "v_movrelsd_2_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                // movrelsd_2 indexes them apart: the source by M0[9:0], the
                // destination by M0[25:16].
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let (si, di) = if name == "v_movrelsd_2_b32" {
                    self.m0_halves(m0)
                } else {
                    (m0, m0)
                };
                let value = if name != "v_movreld_b32" {
                    self.rel_read(inst, sreg, si)?
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = self.old_vgpr(inst, dreg + k)?;
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, di, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swaprel_b32" => {
                // v[dst + M0[25:16]] and v[src + M0[9:0]] trade values. The
                // console writes the destination first and the source last,
                // so a register both name keeps its value.
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative swap");
                };
                let (si, di) = self.m0_halves(m0);
                let to_dst = self.rel_read(inst, sreg, si)?;
                let to_src = self.rel_read(inst, dreg, di)?;
                let sbase = self.s.m.i32(u32::from(sreg));
                let dbase = self.s.m.i32(u32::from(dreg));
                let at_src = self.s.m.bin(c, Bin::Add, sbase, si);
                let at_dst = self.s.m.bin(c, Bin::Add, dbase, di);
                // Both ranges run to the program's last VGPR: their union
                // starts at the lower base.
                let first = sreg.min(dreg);
                for k in 0..=255u8.saturating_sub(first) {
                    let r = first + k;
                    let Some(_) = inst.write(Reg::V(r)) else {
                        break;
                    };
                    let old = self.old_vgpr(inst, r)?;
                    let rr = self.s.m.i32(u32::from(r));
                    let is_dst = self.s.m.cmp(c, pred::EQ, at_dst, rr);
                    let x = self.s.m.select(c, is_dst, to_dst, old);
                    let is_src = self.s.m.cmp(c, pred::EQ, at_src, rr);
                    let x = self.s.m.select(c, is_src, to_src, x);
                    self.write_placed(inst, Reg::V(r), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let x = f(self, 0)?;
                let mut a = x;
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, x);
                    if name == "v_cvt_rpi_i32_f32" {
                        // floor(x + 0.5) of the exact sum, which a float
                        // sum rounds first (0.5 - 2^-25 would give 1, and
                        // 2^23 + 1 the even number above it): one above
                        // the floor from half a step on. x - floor(x) is
                        // exact.
                        let (h, one) = (0.5f32.to_bits(), 1f32.to_bits());
                        let (h, one) = (self.s.m.f32(h), self.s.m.f32(one));
                        let rest = self.s.m.bin(c, Bin::Sub, x, a);
                        let up = self.s.m.cmp(c, pred::OGE, rest, h);
                        let above = self.s.m.bin(c, Bin::Add, a, one);
                        a = self.s.m.select(c, up, above, a);
                    }
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware's two plain ones do (checked on the RX 6700 XT
                // and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let mut r = self.s.m.cast(c, op, a, self.s.t.i32);
                if name == "v_cvt_flr_i32_f32" || name == "v_cvt_rpi_i32_f32" {
                    // These two saturate a NaN too, by its sign (its
                    // exponent is above 30, as that of any other magnitude
                    // too great).
                    let bits = self.as_int(x);
                    let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                    let (abs, inf, zero) = (k(self, 0x7FFF_FFFF), k(self, 0x7F80_0000), k(self, 0));
                    let (max, min) = (k(self, 0x7FFF_FFFF), k(self, 0x8000_0000));
                    let mag = self.s.m.bin(c, Bin::And, bits, abs);
                    let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
                    let neg = self.s.m.cmp(c, pred::SLT, bits, zero);
                    let sat = self.s.m.select(c, neg, min, max);
                    r = self.s.m.select(c, nan, sat, r);
                }
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                // An unavailable old value is zero, as for other partial
                // writes; never resurrect bits from before a full write.
                let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    _ => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pknorm_i16_f32"
            | "v_cvt_pknorm_u16_f32"
            | "v_cvt_pknorm_i16_f16"
            | "v_cvt_pknorm_u16_f16" => {
                // src0 into the low half, src1 into the high half, each
                // clamped and scaled to a 16-bit snorm or unorm.
                let signed = name.contains("_i16_");
                let (a, bb) = if name.ends_with("_f16") {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (f(self, 0)?, f(self, 1)?)
                };
                let lo = self.norm16(a, signed);
                let hi = self.norm16(bb, signed);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond =
                    self.mask_cond(inst, v.src.get(2), c, "a select on an untracked mask")?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    Some(self.mask_cond(
                        inst,
                        v.src.get(2),
                        c,
                        "a carry-in from an untracked mask",
                    )?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                if let Some((base, publish)) = self.logical_lane_sites.get(&(b, i)).copied() {
                    let r = self.logical_lane_read(base, publish, a, lane);
                    return int(self, r);
                }
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                let lane = if self.mask_data == MaskData::Host {
                    self.s.flags |= flags::WAVE_OPS;
                    self.console_lane(lane)
                } else {
                    self.cross_lane(false)?;
                    lane
                };
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                // The lane space is `v_readlane`'s: a host lane, or a
                // console's constant lane of the lane's own wave.
                let lane = if self.mask_data == MaskData::Host {
                    self.s.flags |= flags::WAVE_OPS;
                    self.console_lane(lane)
                } else {
                    self.cross_lane(false)?;
                    lane
                };
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                // The original mesh group input v5 is groupId * mesh_stride
                // for every active lane (io::raise verifies its only use is
                // this readfirstlane). Reading the first lane therefore has
                // exactly the same value as reading the current lane, even
                // when a 64-lane console mesh wave executes on a 32-lane
                // host. This does not certify the program's OTHER wave ops;
                // the host's mesh wave-size refusal stays in place.
                if self.s.mesh.is_some()
                    && let Some(value) = inst.read(Reg::V(5))
                    && self.x.io.inputs.iter().any(|&(v, input)| {
                        v == value && input == crate::io::Input::MeshGroup
                    })
                {
                    return int(self, a);
                }
                let scalar_select = self.s.mesh.is_some() && self.uniform_scalar_select_at_firstlane(inst);
                let uniform_fetch = self.s.mesh.is_some() && self.uniform_buffer_fetch_at_firstlane(inst);
                let uniform_arithmetic = self.s.mesh.is_some()
                    && (self.uniform_arithmetic_firstlane(inst, &mut HashSet::new(), 0)
                        || self.partial_reciprocal_firstlane(inst)
                        || self.partial_data_conversion_firstlane(inst));
                if scalar_select || uniform_fetch || uniform_arithmetic
                    || self.lds_allocation.is_some_and(|p| p.reciprocal == Some((b,i))) {
                    return int(self, a);
                }
                if let Some(&base) = self.mesh_broadcasts.get(&(b, i)) {
                    let r = self.mesh_broadcast(base, a);
                    return int(self, r);
                }
                self.s.flags |= flags::WAVE_OPS;
                let r = if self.mask_data == MaskData::Host && !self.x.native.wave64 {
                    // The first lane of the lane's own console wave: the low
                    // set bit of its half of the exec ballot, renumbered to
                    // a host lane.
                    let exec = inst.read(Reg::S(EXEC)).ok_or(LANE_MASK)?;
                    let d = self.mask_dword(exec)?;
                    let f = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[d]);
                    let base = self.lane_base();
                    let first = self.s.m.bin(c, Bin::Or, f, base);
                    self.read_lane(a, first)
                } else {
                    self.s.op(
                        c,
                        "waveReadLaneFirst.i32",
                        Attr::NoUnwind,
                        self.s.t.i32,
                        118,
                        &[a],
                    )
                };
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.pack64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// Packs low and high 32-bit dwords into an i64.
    fn pack64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ => self.f16_operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve), sign-extended
        // above the low word (sext), or zero (pad; a sign extension above the
        // high word adds nothing, and bits below it pad with zero).
        let mut keep = true;
        let mut sext = false;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4, 1) => (keep, sext) = (false, true),
                (4, 0) | (6, _) => keep = false,
                (5, 0 | 1) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let r = if sext {
            let t = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::AShr, t, k16)
        } else {
            let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                Some(Ok(o)) if keep => o,
                _ => self.s.m.i32(0),
            };
            let x = self.s.m.bin(c, Bin::And, x, low);
            if high {
                let o = self.s.m.bin(c, Bin::And, old, low);
                let x = self.s.m.bin(c, Bin::Shl, x, k16);
                self.s.m.bin(c, Bin::Or, o, x)
            } else {
                let keep = self.s.m.i32(0xFFFF_0000);
                let o = self.s.m.bin(c, Bin::And, old, keep);
                self.s.m.bin(c, Bin::Or, o, x)
            }
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_sub_nc_u32"
            | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() && self.logical_lane_sites.is_empty() {
                    self.cross_lane(false)?;
                }
                // On a host wave wider than the console's the upper half is a
                // second console wave: `lo` counts bits of the lane's own
                // half's mask below its lane in it, and a wave32 mask's high
                // half is empty. The count is a lane number, and lane numbers
                // stay host lanes: a second wave's rank takes its 32-lane
                // base, so `v_mbcnt_lo(-1)` is still the lane index.
                let host_halves = self.mask_data == MaskData::Host && !self.x.native.wave64;
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    if host_halves {
                        let k31 = k(self, 31);
                        self.s.m.bin(c, Bin::And, lane, k31)
                    } else {
                        lane
                    }
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = if host_halves && name == "v_mbcnt_hi_u32_b32" {
                    k(self, 0)
                } else {
                    self.s.m.bin(c, Bin::And, a, m)
                };
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                let n = if host_halves && name == "v_mbcnt_lo_u32_b32" {
                    let base = self.s.m.bin(c, Bin::And, lane, k32);
                    self.s.m.bin(c, Bin::Add, n, base)
                } else {
                    n
                };
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                // Apply abs/neg bitwise so sNaNs are not quieted and denormals
                // are not flushed to zero before classification.
                let (mut x, abs_mask, sign_bit) = if f16 {
                    (self.half_src(inst, 0)?, 0x7FFF, 0x8000)
                } else {
                    let op = *v.src.first().ok_or("a missing operand")?;
                    let raw = self.operand(inst, &op)?;
                    let x = match Self::sdwa_src(inst, 0) {
                        Some((sel, sext)) => self.select_bits(raw, sel, sext),
                        None => raw,
                    };
                    (x, 0x7FFF_FFFF, 0x8000_0000)
                };
                if v.abs & 1 != 0 {
                    let m = self.s.m.i32(abs_mask);
                    x = self.s.m.bin(c, Bin::And, x, m);
                }
                if v.neg & 1 != 0 {
                    let s = self.s.m.i32(sign_bit);
                    x = self.s.m.bin(c, Bin::Xor, x, s);
                }
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                // Always false or true, NaNs included: constants, as for the
                // integer compares. As `fcmp false`/`fcmp true`, which DXC
                // never emits, WARP removes its device at the first dispatch.
                if op == "f" || op == "tru" {
                    self.lane(inst, Some(op == "tru"));
                    return Ok(());
                }
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else {
                    let p = match op {
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                },
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let (id, col) = l
                .find_input(row, ch)
                .ok_or("an input the signature lacks")?;
            let e = &mut l.s.inputs[id];
            if e.interp != mode && e.usage != 0 {
                return Err("one input interpolated two ways");
            }
            e.interp = mode;
            match l.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| {
                    let b = if i.row2 != i.row {
                        Some(i.row2)
                    } else if i.pack2 {
                        half_input_partners(&self.x.native.io, &self.s.inputs)
                            .get(&i.row)
                            .copied()
                    } else {
                        None
                    };
                    (i.pack2, b, i.pack2 && i.row2 != i.row)
                }),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => channel(self, attribute.row, attribute.channel)?,
            Some((pack2, b, explicit_second_row)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 && f16 {
                        packed_input_half(&l.s.inputs, attribute.row, b, h, explicit_second_row)?
                    } else if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((row, ch)) = src else {
                        if f16 && explicit_second_row {
                            return Ok(Some(l.s.m.i32(0)));
                        }
                        return Ok(None);
                    };
                    let x = channel(l, row, ch)?;
                    let f = l.as_float(x);
                    let i32t = l.s.t.i32;
                    Ok(Some(l.s.op(
                        l.cur,
                        "legacyF32ToF16",
                        Attr::ReadNone,
                        i32t,
                        130,
                        &[f],
                    )))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    let lo = half(self, h + u8::from(attribute.high))?
                        .ok_or("a packed input half no element holds")?;
                    self.low_half(inst, reg, lo)
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let z = self.s.m.i32(0);
                    let lo = half(self, h)?.unwrap_or(z);
                    let hi = half(self, h + 1)?.unwrap_or(z);
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if *a16 && name.starts_with("image_sample_b") {
            return Err("a biased image sample with 16-bit coordinates");
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first.
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs
                .get(if *a16 { k / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (1/3 for 32-bit, 3/0xf for 64-bit;
            // cmpswap takes twice the value's).
            let cmpswap = code == u32::MAX;
            let wide = match (*dmask, cmpswap) {
                (1, false) | (3, true) => false,
                (3, false) | (0xF, true) => true,
                _ => return Err("an image atomic of unsupported width"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let dwords = match (wide, cmpswap) {
                (false, false) => 1,
                (false, true) | (true, false) => 2,
                (true, true) => 4,
            };
            let vals = self.vgprs(inst, data, dwords)?;
            let (ty, comp, sfx) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_ON_TYPED_RESOURCE;
                (self.s.m.int(64), 0x107, "i64")
            } else {
                (self.s.t.i32, 0x105, "i32")
            };
            let mut value = if wide {
                self.pack64(vals[0], vals[1])
            } else {
                vals[0]
            };
            if op == "sub" {
                let z = if wide {
                    self.s.m.int_const(ty, 0)
                } else {
                    self.s.m.i32(0)
                };
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let old = if wide {
                match (
                    cache.glc,
                    inst.read(Reg::V(reg)),
                    inst.read(Reg::V(reg + 1)),
                ) {
                    (true, Some(lo), Some(hi)) => {
                        let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
                        self.pack64(lo, hi)
                    }
                    _ => self.s.m.int_const(ty, 0),
                }
            } else {
                match inst.read(Reg::V(reg)) {
                    Some(v) if cache.glc => self.i32_of(v)?,
                    _ => self.s.m.i32(0),
                }
            };
            let h = self.handle(site, src, false, (kind | 0x1000, comp))?;
            // Typed image atomics use ResourceDescriptorHeap handles; DXIL
            // requires the separate heap-resource int64 atomic feature bit.
            if wide {
                self.s.flags |= flags::ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE;
            }
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            let r = if cmpswap {
                // cmpswap: data is (new, compare).
                let cmp = if wide {
                    self.pack64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([cmp, value]);
                self.s.op(
                    self.cur,
                    &format!("atomicCompareExchange.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s.op(
                    self.cur,
                    &format!("atomicBinOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    78,
                    &args,
                )
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if cache.glc {
                if wide {
                    let (lo, hi) = self.split64(r);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), r, None);
                }
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            if *dmask != 0xF {
                return Err("an image store of some channels");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for k in 0..4u8 {
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let (op, sample) = match name {
            "image_sample" | "image_sample_o" | "image_sample_cl" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" | "image_sample_b_cl" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            let first =
                usize::from(offset_word) + usize::from(name == "image_sample_b") + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            let grads = if op == 63 { 2 * spatial } else { 0 };
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates and LOD are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_val(self, first + grads + k)?;
                *slot = to_float(self, x);
            }
            if kind == 5 {
                if *unorm || op == 63 || op == 81 {
                    return Err("a cube sample with texel coordinates, gradients or a level query");
                }
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            let clamp = if name.contains("_cl") {
                let x = addr_val(self, first + grads + coords)?;
                to_float(self, x)
            } else {
                fu
            };
            match op {
                60 => {
                    if offset_word {
                        // DXIL takes immediate offsets in -8..=7; the
                        // console's are 6-bit and may be computed.
                        let w = addr_val(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = addr_val(self, 0)?;
                    let bias = self.as_float(bias);
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        let x = addr_val(self, first + k)?;
                        g[k] = to_float(self, x);
                        let y = addr_val(self, first + spatial + k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_val(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(fu);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_val(self, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        let w = addr_val(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    self.s.op(
                        c,
                        "textureGather.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        73,
                        &args,
                    )
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || matches!(name, "buffer_load_ushort" | "buffer_load_sshort" | "buffer_store_short") => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let mut index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if !typed && used.is_none() {
            if let Some(element) = index {
                if let Some(stride) = self.slot_stride(src)? {
                    let at = self.s.m.bin(c, Bin::Mul, element, stride);
                    let byte = self.s.m.bin(c, Bin::Add, at, off);
                    // An unwritten/wrong-type heap slot has no meaningful
                    // stride. Force a raw OOB access rather than silently
                    // redirecting every indexed access to element zero.
                    let zero = self.s.m.i32(0);
                    let invalid = self.s.m.i32(u32::MAX);
                    let missing = self.s.m.cmp(c, pred::EQ, stride, zero);
                    off = self.s.m.select(c, missing, invalid, byte);
                    index = None;
                }
            }
        }
        // The original used-resource record supplies a known byte stride.
        // Access through a raw descriptor at the computed byte offset so
        // indexed and byte-addressed views work the same way on NVIDIA.
        if let (Some(element), Some(u)) = (index, used)
            && !typed && u.kind != ResourceType::Cbv
            && u.stride >= 4 && u.stride.is_multiple_of(4)
        {
            let stride = self.s.m.i32(u.stride);
            let at = self.s.m.bin(c, Bin::Mul, element, stride);
            off = self.s.m.bin(c, Bin::Add, at, off);
            index = None;
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if let Some(&entry) = self.static_cbv_sites.get(&site) {
                for k in 0..count {
                    let four = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, byte, four);
                    let x = self.static_cbv_dword(entry, a);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let size = self.root_view_size(src);
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let (a, ok) = match size {
                    Some(size) => {
                        let (ok, a) = self.in_root_view(size, a, 4);
                        (a, Some(ok))
                    }
                    None => (a, None),
                };
                let x = self.cbuffer_dword(h, a);
                let x = self.zero_unless(ok, x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if matches!(name, "buffer_load_ushort" | "buffer_load_sshort") {
            // Halfword reads use a raw byte address. The containing dword
            // is read without alignment assumptions, then exactly the low
            // or high 16 bits are selected. A halfword at byte 3 crosses a
            // dword boundary, so it needs the following dword as well.
            if index.is_some() { return Err("a 16-bit buffer load by element index"); }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let k3 = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, off, k3);
            let aligned = self.s.m.bin(c, Bin::Sub, off, byte);
            let root_size = self.root_view_size(src);
            let three = self.s.m.i32(3);
            let eight = self.s.m.i32(8);
            let shift = self.s.m.bin(c, Bin::Shl, byte, three);
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(c, Bin::Add, aligned, four);
            let mask = self.s.m.i8(1);
            let align = self.s.m.i32(4);
            let load = |l: &mut Self, a: V| -> V {
                let (a, ok) = if let Some(size) = root_size {
                    let (ok, safe) = l.in_root_view(size, a, 4);
                    (safe, Some(ok))
                } else { (a, None) };
                let r = l.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, l.s.t.res_ret,
                    139, &[h, a, iu, mask, align]);
                let x = l.s.m.extract(c, r, 0);
                let x = l.as_int(x);
                l.zero_unless(ok, x)
            };
            let low = load(self, aligned);
            let mut x = self.s.m.bin(c, Bin::LShr, low, shift);
            let high = load(self, next);
            let high = self.s.m.bin(c, Bin::Shl, high, eight);
            let cross = self.s.m.cmp(c, pred::EQ, byte, k3);
            let zero = self.s.m.i32(0);
            let high = self.s.m.select(c, cross, high, zero);
            x = self.s.m.bin(c, Bin::Or, x, high);
            let mask16 = self.s.m.i32(0xffff);
            x = self.s.m.bin(c, Bin::And, x, mask16);
            if name == "buffer_load_sshort" {
                let k16 = self.s.m.i32(16);
                x = self.s.m.bin(c, Bin::Shl, x, k16);
                x = self.s.m.bin(c, Bin::AShr, x, k16);
            }
            if let Some(Operand::Vector { reg, .. }) = *data {
                self.write(inst, Reg::V(reg), x, guard);
                return Ok(());
            }
            return Err("a 16-bit buffer load destination");
        }
        if name == "buffer_store_short" {
            if index.is_some() { return Err("a 16-bit buffer store by element index"); }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let three = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let aligned = self.s.m.bin(c, Bin::Sub, off, byte);
            let root_size = self.root_view_size(src);
            let eight = self.s.m.i32(8);
            let shift = self.s.m.bin(c, Bin::Shl, byte, eight);
            let mask = self.s.m.i32(0xffff);
            let data = self.vgprs(inst, data.as_ref().ok_or("a 16-bit buffer store value")?, 1)?[0];
            let value = self.s.m.bin(c, Bin::And, data, mask);
            let value = self.s.m.bin(c, Bin::Shl, value, shift);
            let clear = self.s.m.bin(c, Bin::Shl, mask, shift);
            let ones = self.s.m.i32(u32::MAX);
            let keep = self.s.m.bin(c, Bin::Xor, clear, ones);
            let and_op = self.s.m.i32(1);
            let or_op = self.s.m.i32(2);
            // A halfword write must have both bytes. Never partially update
            // a root view when its second byte is beyond its declared end.
            let primary_ok = root_size.map(|size| self.in_root_view(size, off, 2).0);
            let active = self.narrow(guard, primary_ok);
            let at = self.begin_guarded(active);
            for (op, x) in [(and_op, keep), (or_op, value)] {
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind,
                    self.s.t.i32, 78, &[h, op, aligned, iu, iu, x]);
            }
            // Rare unaligned stores can straddle a dword: preserve both
            // neighboring words. The original compiler normally produces
            // even offsets for a halfword value.
            let crossing = self.s.m.cmp(self.cur, pred::EQ, byte, three);
            let secondary = root_size.map(|size| {
                let four = self.s.m.i32(4);
                let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
                self.in_root_view(size, next, 4).0
            });
            let active = self.narrow(Some(crossing), secondary);
            let outer = self.begin_guarded(active);
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let high = self.s.m.bin(self.cur, Bin::LShr, data, eight);
            let low8 = self.s.m.i32(0xff);
            let high = self.s.m.bin(self.cur, Bin::And, high, low8);
            let keep_next = self.s.m.i32(0xffff_ff00);
            for (op, x) in [(and_op, keep_next), (or_op, high)] {
                self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind,
                    self.s.t.i32, 78, &[h, op, next, iu, iu, x]);
            }
            self.end_guarded(outer, &[]);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Root views: the bytes accessed checked against the view's size
        // (raw accesses, and the raw loads formatted ones make).
        let (off, root_ok) = match self.root_view_size(src) {
            Some(size) if !typed || format.is_some() => {
                let bytes = if format.is_some() { 16 } else { 4 * count };
                let (ok, at) = self.in_root_view(size, off, bytes);
                (at, Some(ok))
            }
            _ => (off, None),
        };
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    (h, [index, off, iu])
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let access = self.narrow(guard, root_ok);
            let at = self.begin_guarded(access);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride < 4 || !stride.is_multiple_of(4) {
                            return Err("a structured buffer of unknown stride");
                        }
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let access = self.narrow(guard, root_ok);
            let at = self.begin_guarded(access);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.format_load(h, off, f, count)?.into_iter().enumerate() {
                let x = self.zero_unless(root_ok, x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride < 4 || !stride.is_multiple_of(4) {
                        return Err("a structured buffer of unknown stride");
                    }
                    let h = self.handle(site, src, false, (12 | u, stride))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, index, off, mask, align],
                    )
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            let x = self.zero_unless(root_ok, x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, None));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports;
        // guard against threads past the declared vertex/primitive maximum.
        let at = if mesh {
            let m = self.s.mesh.expect("mesh");
            let threads = self.s.threads.expect("mesh")[0];
            let limit = if primitive {
                m.max_primitives
            } else {
                m.max_vertices
            };
            let guard = if threads > limit {
                let flat = self.mesh_thread();
                let lim = self.s.m.i32(limit);
                let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, lim);
                Some(match guard {
                    Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
                    None => in_bounds,
                })
            } else {
                guard
            };
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area); any vertex index at or past `max_vertices` is
    /// also culled so out-of-range indices never fault the host rasterizer.
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let mesh = self.s.mesh.expect("mesh");
        let threads = self.s.threads.expect("mesh")[0];
        let flat = self.mesh_thread();
        let guard = if threads > mesh.max_primitives {
            let mp = self.s.m.i32(mesh.max_primitives);
            let in_bounds = self.s.m.cmp(self.cur, pred::IULT, flat, mp);
            Some(match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
                None => in_bounds,
            })
        } else {
            guard
        };
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let mv = self.s.m.i32(mesh.max_vertices);
        let mut bad = self.s.m.cmp(c, pred::SLT, x, z);
        let mut idxs = [z; 3];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            let oob = self.s.m.cmp(c, pred::IUGE, i, mv);
            bad = self.s.m.bin(c, Bin::Or, bad, oob);
            idxs[k as usize] = i;
        }
        let mut args = vec![flat];
        for i in idxs {
            args.push(self.s.m.select(c, bad, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Branch { taken, not_taken, .. }
                if self.relative_waterfalls.contains_key(&b) => {
                debug_assert_eq!(taken, b);
                self.enter_loop(b, not_taken, self.cur);
                Term::Br(self.block[not_taken].expect("certified waterfall successor"))
            }
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    if self.lds_allocation.is_some() {
                        let no = self.s.m.bool(false);
                        let capped = self.s.m.cmp(self.cur, pred::EQ, ok, no);
                        let failed = self.s.m.bin(self.cur, Bin::And, cond, capped);
                        self.allocation_cap_edge(b, taken, failed);
                    }
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    if self.lds_allocation.is_some() {
                        let no = self.s.m.bool(false);
                        let back = self.s.m.cmp(self.cur, pred::EQ, cond, no);
                        let failed = self.s.m.bin(self.cur, Bin::And, back, stop);
                        self.allocation_cap_edge(b, not_taken, failed);
                    }
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = super::loop_limit();
        #[cfg(test)]
        let limit = if self.lds_allocation.is_some() {
            ALLOCATION_TEST_LIMIT.with(|c| c.get().unwrap_or(limit))
        } else { limit };
        let limit = self.s.m.i32(limit);
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn allocation_cap_edge(&mut self, b: usize, target: usize, failed: V) {
        let abort = self.abort_block();
        let go = self.s.m.new_block();
        let old = self.cur;
        self.s.m.terminate(old, Term::CondBr(failed, abort, go));
        self.cur = go;
        self.tail[b] = Some(go);
        let last = self.counters.get_mut(&target).unwrap().1.last_mut().unwrap();
        assert_eq!(last.1, old);
        last.1 = go;
    }

    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if let Some(plan) = self.lds_allocation {
            let old = self.cur;
            self.cur = a;
            let ptr = self.allocation_abort_ptr(plan.abort_slot);
            let one = self.s.m.i32(1);
            self.s.m.atomic(a, super::bitcode::Rmw::Or, ptr, one);
            self.s.m.terminate(a, Term::Br(self.block[plan.gate].expect("certified gate")));
            self.cur = old;
            self.abort = Some(a);
            return a;
        }
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        // A phi that only another read phi consumes is live too. Without this, a value with no DXIL form that
        // reaches a read through two joins is taken for dead at the first join and replaced by zero below, and the
        // lowering accepts what it refuses when the same value is read directly.
        let mut work: Vec<Value> = self.read_phis.iter().copied().collect();
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &arg in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&arg) && self.read_phis.insert(arg) {
                    work.push(arg);
                }
            }
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) {
                    self.materialize_barycentric(arg);
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            if self.lds_allocation.is_some_and(|p| p.gate == b) {
                if let Some(a) = self.abort {
                    let dummy = if bit { self.s.m.bool(false) } else { self.s.m.i32(0) };
                    incoming.push((dummy, a));
                }
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let m =
                    *self.x.st.masks.get(&phi.args[n]).ok_or_else(|| {
                        unsupported(offset, "a lane-mask join of an untracked mask")
                    })?;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            if self.lds_allocation.is_some_and(|p| p.gate == b) {
                if let Some(a) = self.abort { incoming.push((self.s.m.bool(false), a)); }
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }
}

/// IR blocks containing `s_barrier` (not already in `skip`) that are
/// control-dependent on a non-uniform (per-lane) branch: one path from a
/// divergent branch can reach the barrier while the other can reach an exit
/// without passing through it. Emitting `SYNC_THREAD_GROUP` in such a block
/// deadlocks 32-lane hosts whenever warps diverge.
fn once_only_rendezvous(exits: &[LaneExit], target: usize) -> bool {
    if target >= exits.len() { return false; }
    fn successors(exit: LaneExit) -> Vec<usize> {
        match exit {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            _ => vec![],
        }
    }
    // Iterative DFS, with the candidate absorbing. Every path must reach it;
    // a cycle even with a possible exit is deliberately rejected.
    let mut color = vec![0u8; exits.len()];
    let mut stack = vec![(0usize, false)];
    let mut reached = false;
    while let Some((b, leaving)) = stack.pop() {
        if b >= exits.len() { return false; }
        if b == target { reached = true; continue; }
        if leaving { color[b] = 2; continue; }
        if color[b] == 1 { return false; }
        if color[b] == 2 { continue; }
        let next = successors(exits[b]);
        if next.is_empty() { return false; }
        color[b] = 1;
        stack.push((b, true));
        stack.extend(next.into_iter().map(|t| (t, false)));
    }
    if !reached { return false; }
    // A later revisit would invalidate the dedicated-slot, once-only fence.
    let mut seen = HashSet::new();
    let mut todo = successors(exits[target]);
    while let Some(b) = todo.pop() {
        if b >= exits.len() || b == target { return false; }
        if seen.insert(b) { todo.extend(successors(exits[b])); }
    }
    true
}

fn divergent_barriers(
    p: &crate::ir::Program,
    st: &crate::structure::Structured,
    skip: &HashSet<usize>,
) -> HashSet<usize> {
    let n = p.blocks.len();
    let barriers: Vec<usize> = (0..n)
        .filter(|b| !skip.contains(b) && p.blocks[*b].insts.iter().any(|i| i.name() == "s_barrier"))
        .collect();
    if barriers.is_empty() {
        return HashSet::new();
    }
    let divergent: Vec<(usize, usize)> = (0..n)
        .filter_map(|u| match st.exits[u] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } if taken != not_taken
                && st
                    .bools
                    .support(when)
                    .iter()
                    .any(|a| !matches!(a, Atom::Uniform(_))) =>
            {
                Some((taken, not_taken))
            }
            _ => None,
        })
        .collect();
    if divergent.is_empty() {
        return HashSet::new();
    }
    let ends: Vec<usize> = (0..n)
        .filter(|&b| matches!(st.exits[b], LaneExit::End))
        .collect();
    let mut out = HashSet::new();
    for b in barriers {
        let mut can_reach = vec![false; n];
        let mut stack = vec![b];
        can_reach[b] = true;
        while let Some(cur) = stack.pop() {
            for &pred in &p.blocks[cur].preds {
                if !can_reach[pred] {
                    can_reach[pred] = true;
                    stack.push(pred);
                }
            }
        }
        let mut can_bypass = vec![false; n];
        for &e in &ends {
            if e != b && !can_bypass[e] {
                can_bypass[e] = true;
                stack.push(e);
            }
        }
        while let Some(cur) = stack.pop() {
            for &pred in &p.blocks[cur].preds {
                if pred != b && !can_bypass[pred] {
                    can_bypass[pred] = true;
                    stack.push(pred);
                }
            }
        }
        if divergent
            .iter()
            .any(|&(t, f)| (can_reach[t] && can_bypass[f]) || (can_reach[f] && can_bypass[t]))
        {
            out.insert(b);
        }
    }
    out
}

#[cfg(test)]
#[path = "lower_tests.rs"]
mod tests;
