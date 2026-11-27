//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and denormal
//! operands of `v_mul_legacy_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}
/// VOP1 SDWA `op_sdwa vdst, vsrc0 dst_sel dst_unused src0_sel`.
const fn vop1_sdwa(
    op: u16,
    vdst: u8,
    vsrc0: u8,
    dst_sel: u8,
    dst_unused: u8,
    src0_sel: u8,
) -> [u32; 2] {
    [
        vop1(op, vdst, 0xF9),
        vsrc0 as u32
            | (dst_sel as u32 & 7) << 8
            | (dst_unused as u32 & 3) << 11
            | (src0_sel as u32 & 7) << 16,
    ]
}

/// VOP2 SDWA `op_sdwa vdst, vsrc0, vsrc1 dst_sel dst_unused src0_sel src1_sel`.
const fn vop2_sdwa(
    op: u8,
    vdst: u8,
    vsrc0: u8,
    vsrc1: u8,
    dst: (u8, u8),
    src_sel: [u8; 2],
) -> [u32; 2] {
    [
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | 0xF9,
        vsrc0 as u32
            | (dst.0 as u32 & 7) << 8
            | (dst.1 as u32 & 3) << 11
            | (src_sel[0] as u32 & 7) << 16
            | (src_sel[1] as u32 & 7) << 24,
    ]
}


/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `ds_swizzle_b32 vdst, vsrc offset:offset`.
const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0x35 << 18 | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, voff, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, voff: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | voff as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_valu_cs_vgprs(body, 8)
}

fn build_valu_cs_vgprs(body: &[u32], vgprs: u16) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    build_cs_group_vgprs(&words, [1, 1, 1], false, 0, false, vgprs)
}

/// Lowers a synthetic RDNA2 compute shader of `threads` threads in one row
/// (thread id in `v0`): 4 dwords of root constants in `s0..s3`, the
/// descriptor table address in `s4`, a raw UAV at heap index 0.
fn build_cs(words: &[u32], threads: u32) -> HostShader {
    build_cs_config(words, threads, false, 0, false)
}

fn build_cs_config(words: &[u32], threads: u32, wave64: bool, lds_bytes: u32, expect_allocation: bool) -> HostShader {
    build_cs_group(words, [threads, 1, 1], wave64, lds_bytes, expect_allocation)
}

fn build_cs_group(words: &[u32], threads: [u32; 3], wave64: bool, lds_bytes: u32, expect_allocation: bool) -> HostShader {
    build_cs_group_vgprs(words, threads, wave64, lds_bytes, expect_allocation, 8)
}

fn build_cs_group_vgprs(words: &[u32], threads: [u32; 3], wave64: bool, lds_bytes: u32, expect_allocation: bool, vgprs: u16) -> HostShader {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15)
                | (if threads[2] > 1 { 2 } else if threads[1] > 1 { 1 } else { 0 }) << 11,
            threads,
        }),
        code,
    };

    lower_native_cs(&native, expect_allocation)
}

fn lower_native_cs(native: &NativeProgram, expect_allocation: bool) -> HostShader {
    lower_native_cs_options(native, expect_allocation, false)
}

fn lower_native_cs_options(native: &NativeProgram, expect_allocation: bool, static_cbvs: bool) -> HostShader {
    let (p, st) = crate::range::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = st.expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let parts = lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    if expect_allocation {
        assert!(lower::lds_allocation_audit(&parts).is_some(), "allocator certificate rejected synthetic shader");
    }
    let lowered = lower::lower_with_static_cbvs(&parts, static_cbvs).expect("lower");
    if expect_allocation {
        assert!(!lowered.wave_ops);
        let barriers = lowered.insts.iter().filter(|i| i.contains("k80,")).count();
        assert_eq!(barriers, 3);
    }

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        static_table_cbvs: lowered.static_table_cbvs,
        heap_buffer_classes: lowered.heap_buffer_classes,
        heap_demands: lowered.heap_demands,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

#[test]
fn static_table_cbv_packed_numerical() {
    // Two exact static CBVs and an ordinary raw UAV in the same table.
    let mut w = vec![s_mov_b32(5,128)];
    for (reg, offset) in [(8,0),(12,32),(16,64)] {
        let mut load = s_load_dwordx4(reg,4);
        load[1] |= offset;
        w.extend(load);
    }
    // Scalar x4 crosses the first CBV's boundary; second CBV uses a
    // dynamic scalar address and a two-dword per-thread vector read.
    w.extend([0xF4280504, 0xFA00000C]); // s[20:23] from s[8:11]+12
    w.extend([0xF4200606, 0x00000000]); // s24 from s[12:15]+s0
    w.push(vop2(0x1A,1,130,0)); // v1=tid*4
    w.extend([0xE0341000,0x80030201]); // buffer_load_dwordx2 v[2:3],v1,s[12:15],0 offen
    w.push(vop2(0x1A,4,133,0)); // v4=tid*32
    for k in 0..7u8 {
        let data = if k < 5 { w.push(vop1(1,5,sgpr(20+k))); 5 } else { k-3 };
        let mut store = buffer_store_dword_offen(data,4,16);
        store[0] |= u32::from(k)*4;
        w.extend(store);
    }
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 8,
        layout: vec![
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
        ],
        used: [ResourceType::Cbv, ResourceType::Cbv, ResourceType::Uav].into_iter().enumerate().map(|(k,kind)| UsedResource {
            kind, root: RootType::Table, table_offset_dwords: 8*k as u32, lower_bound: 0,
            stride: 0, dwords: 4, user_data: 4, slot: 1,
        }).collect(),
        io: StageIo::Cs(ComputeIo { rsrc2: 5<<1, threads: [8,1,1] }),
        code: w.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let shader = lower_native_cs_options(&native,false,true);
    assert_eq!(shader.static_table_cbvs, [super::StaticTableCbv {slot:1,table_offset_dwords:0}, super::StaticTableCbv {slot:1,table_offset_dwords:8}]);
    assert_eq!(shader.root_srvs, [super::STATIC_CBV_SCRATCH]);
    assert!(lower_native_cs(&native,false).static_table_cbvs.is_empty());
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware",false),("warp",true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root,pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: {e}"));
        let data = [0x7fc12345,0x80000000,0x7fa65432,123,456,789,987,0xffffffff];
        for (offset, null) in [(0,false),(28,false),(32,false),(u32::MAX,false),(0,true)] {
            let mut packed = vec![0xBADBADu32;192];
            packed[..5].copy_from_slice(&[256,16,if null {16} else {512},if null {0} else {32},0]);
            packed[64..68].copy_from_slice(&[11,22,33,44]);
            packed[128..136].copy_from_slice(&data);
            let mut ud = vec![0u32;shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = if u.root == RootType::Constants && u.dword == 0 {offset} else {0};
            }
            let result = host.dispatch(&root,&pso,&Dispatch {
                roots:vec![(0,Root::Constants(ud)),(1,Root::View {srv:true,data:packed})],
                // Heap CBV slots contain unrelated bytes; the optimized
                // shader must never read them, nor move the retained UAV.
                views:&[View::Cbv(&[999;4]),View::Cbv(&[888;4]),View::RawUav(64)],samplers:&[],groups:[1,1,1],
            }).unwrap_or_else(|e| panic!("{label}: {e}"));
            let scalar = if null {0} else {data.get((offset/4) as usize).copied().unwrap_or(0)};
            for lane in 0..8 {
                let first = if null {0} else {data[lane]};
                let second = if null {0} else {data.get(lane+1).copied().unwrap_or(0)};
                assert_eq!(&result[0][lane*8..lane*8+8],&[44,0,0,0,scalar,first,second,0],"{label} lane={lane} offset={offset} null={null}");
            }
        }
        ran += 1;
    }
    assert!(ran>0);
}

/// Actual GFX10 LDS byte/halfword opcodes, executed on the hardware adapter
/// and WARP, with poisoned neighboring halves and signed/unsigned values.
/// Each lane writes its own dword, then reads the same address, so this test
/// does not assume the unsupported native 64-lane cross-thread tile ordering.
#[test]
fn native_lds_narrow_reads_numerical() {
    // The second dword encodes vdst:vaddr. DS offsets are byte offsets.
    let samples: [(u32, u32, bool, u32); 12] = [
        (0x3c, 0, false, 16), (0x3c, 2, false, 16),
        (0x3b, 0, true, 16), (0x3b, 2, true, 16),
        (0x3a, 0, false, 8), (0x3a, 1, false, 8),
        (0x3a, 2, false, 8), (0x3a, 3, false, 8),
        (0x39, 0, true, 8), (0x39, 1, true, 8),
        (0x39, 2, true, 8), (0x39, 3, true, 8),
    ];
    let mut words = vec![s_mov_b32(5, 128)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x1a, 2, 130, 0)); // v2 = thread ID * 4
    words.push(vop1(1, 3, sgpr(0)));
    words.push(vop2(0x25, 3, vgpr(0), 3)); // distinct dword for each lane
    words.extend([0xD8340000, 0x00000302]); // ds_write_b32 v2,v3
    words.push(vop2(0x1a, 5, 134, 0)); // per-lane output stride 64 bytes
    for (slot, &(opcode, offset, _, _)) in samples.iter().enumerate() {
        words.extend([0xD8000000 | opcode << 18 | offset, 0x04000002]);
        let mut store = buffer_store_dword_offen(4, 5, 8);
        store[0] |= (slot * 4) as u32;
        words.extend(store);
    }
    words.push(vop2(0x25, 6, 130, 2)); // v6 = v2 + 2: dynamic address
    words.extend([0xD8F00000, 0x04000006]); // u16, offset0, v6
    let mut store = buffer_store_dword_offen(4, 5, 8);
    store[0] |= 12 * 4;
    words.extend(store);
    let mut store = buffer_store_dword_offen(3, 5, 8);
    store[0] |= 13 * 4;
    words.extend(store);
    words.extend([0xD8D80000, 0x04000002]); // plain ds_read_b32
    let mut store = buffer_store_dword_offen(4, 5, 8);
    store[0] |= 14 * 4;
    words.extend(store);
    words.push(S_ENDPGM);
    // Console LDS allocation is in 512-byte granules.
    let shader = build_cs_config(&words, 32, false, 512, false);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) if warp => { eprintln!("WARP lacks narrow LDS pipeline support: {e}"); continue },
            Err(e) => panic!("{label}: narrow LDS compute PSO: {e}"),
        };
        for source in [0x80fe7f81u32, 0x7f018000, 0xff0080ff, 0x0000ffff] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) if u.dword == 0 => source,
                    (RootType::Constants, 0) => 0,
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected narrow LDS root {other:?}"),
                };
            }
            let out = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32 * 16)], samplers: &[], groups: [1, 1, 1],
            }).unwrap_or_else(|e| panic!("{label}: narrow LDS dispatch: {e}"));
            for lane in 0..32 {
                let value = source.wrapping_add(lane);
                for (slot, &(_, byte, signed, bits)) in samples.iter().enumerate() {
                    let x = value >> (8 * byte);
                    let expected = match (bits, signed) {
                        (8, true) => (x as u8 as i8 as i32) as u32,
                        (8, false) => x & 0xff,
                        (16, true) => (x as u16 as i16 as i32) as u32,
                        (16, false) => x & 0xffff,
                        _ => unreachable!(),
                    };
                    assert_eq!(out[0][lane as usize * 16 + slot], expected,
                        "{label}: lane={lane} source={source:#x} slot={slot}");
                }
                assert_eq!(out[0][lane as usize * 16 + 12], (value >> 16) & 0xffff,
                    "{label}: dynamic LDS byte address");
                assert_eq!(out[0][lane as usize * 16 + 13], value,
                    "{label}: original source word was not preserved");
                assert_eq!(out[0][lane as usize * 16 + 14], value,
                    "{label}: full LDS word was not preserved");
                assert!(out[0][lane as usize * 16 + 15..lane as usize * 16 + 16]
                    .iter().all(|&x| x == 0), "{label}: overwritten neighboring output");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "NVIDIA/WARP numerical device required");
}

/// A native halfword store must preserve the neighboring 16 bits of the
/// underlying dword. The actual scene uses this to build an LDS image tile,
/// with separate lanes owning adjacent halfwords.
#[test]
fn native_lds_halfword_writes_preserve_other_half() {
    let mut words = vec![s_mov_b32(5, 128)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x1a, 2, 130, 0)); // each thread's LDS dword
    words.push(vop1(1, 3, sgpr(0)));
    words.push(vop2(0x25, 3, vgpr(0), 3));
    words.extend([0xD8340000, 0x00000302]); // full original dword
    words.push(vop2(0x1a, 5, 132, 0)); // output stride 16 bytes
    words.push(vop1(1, 4, sgpr(1)));
    words.extend([0xD87C0002, 0x00000402]); // ds_write_b16 v2,v4 offset2
    words.extend([0xD8D80000, 0x04000002]); // read full word
    words.extend(buffer_store_dword_offen(4, 5, 8));
    words.push(vop1(1, 4, sgpr(2)));
    words.extend([0xD87C0000, 0x00000402]); // ds_write_b16 v2,v4 offset0
    words.extend([0xD8D80000, 0x04000002]);
    let mut store = buffer_store_dword_offen(4, 5, 8);
    store[0] |= 4;
    words.extend(store);
    let mut store = buffer_store_dword_offen(3, 5, 8);
    store[0] |= 8;
    words.extend(store);
    words.push(S_ENDPGM);
    let shader = build_cs_config(&words, 32, false, 512, false);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) if warp => { eprintln!("WARP lacks LDS halfword pipeline support: {e}"); continue },
            Err(e) => panic!("{label}: halfword LDS PSO: {e}"),
        };
        for (original, high, low) in [
            (0x1122_3344u32, 0x5566_abcd, 0x7788_9911),
            (0xffff_0000, 0x0000_8001, 0x0000_00fe),
            (0x80fe_7f81, 0xabcd_ff00, 0x1234_beef),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => match u.dword {0=>original, 1=>high, 2=>low, _=>0},
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected LDS halfword root {other:?}"),
                };
            }
            let out = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32 * 4)], samplers: &[], groups: [1, 1, 1],
            }).unwrap_or_else(|e| panic!("{label}: LDS halfword dispatch: {e}"));
            for lane in 0..32usize {
                let word = original.wrapping_add(lane as u32);
                let after_high = word & 0xffff | ((high & 0xffff) << 16);
                let after_low = after_high & 0xffff_0000 | (low & 0xffff);
                assert_eq!(&out[0][lane * 4..lane * 4 + 4],
                    &[after_high, after_low, word, 0],
                    "{label}: lane={lane}, halfword stores must preserve the other half");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "NVIDIA/WARP LDS halfword device required");
}

/// Two neighboring lanes write opposite halves of the same shared dword,
/// then the whole 64-thread group observes it after a real group barrier.
/// This catches lost writes from non-atomic whole-dword emulation and tests
/// the two-host-wave interleaving that the original wave64 tile exposes.
#[test]
fn native_lds_adjacent_halfword_writes_survive_cross_wave_barrier() {
    let mut words = vec![s_mov_b32(5, 128)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x1a, 2, 129, 0)); // byte address = thread ID * 2
    words.push(vop1(1, 3, sgpr(0)));
    words.push(vop2(0x25, 3, vgpr(0), 3)); // one distinct halfword per lane
    words.extend([0xD87C0000, 0x00000302]); // ds_write_b16 v2,v3
    words.push(0xBF8A0000); // s_barrier: whole group after both waves wrote
    words.extend([0xD8D80000, 0x04000002]); // ds_read_b32 v4,v2
    words.push(vop2(0x1a, 5, 130, 0)); // output dword for each lane
    words.extend(buffer_store_dword_offen(4, 5, 8));
    words.push(S_ENDPGM);
    let shader = build_cs_config(&words, 64, false, 512, false);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) if warp => { eprintln!("WARP lacks cross-wave LDS pipeline: {e}"); continue },
            Err(e) => panic!("{label}: LDS group barrier pipeline: {e}"),
        };
        for initial in [0x80fe7f81u32, 0x7fff_ffff, 0xffff_0000] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) if u.dword == 0 => initial,
                    (RootType::Constants, 0) => 0,
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected group LDS root {other:?}"),
                };
            }
            let out = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(64)], samplers: &[], groups: [1, 1, 1],
            }).unwrap_or_else(|e| panic!("{label}: halfword group dispatch: {e}"));
            for lane in 0..64usize {
                let pair = (lane / 2) as u32 * 2;
                let lo = initial.wrapping_add(pair) & 0xffff;
                let hi = initial.wrapping_add(pair + 1) & 0xffff;
                assert_eq!(out[0][lane], lo | (hi << 16),
                    "{label}: lane={lane} after group LDS rendezvous");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "a hardware LDS group-barrier device is required");
}

/// The native 8x8 filter's two center planes, four-column halo, and
/// two-iteration symmetric six-pair LDS consumer. Integer data keeps the
/// expected answer exact; the native filter's address proof is exercised.
#[test]
fn certified_filter_lds_numerical() {
    let mut w = vec![s_mov_b32(10, 4), s_mov_b32(11, 128)];
    // s[8:15] is the output 2D UAV descriptor, s5/s6 are group X/Y.
    w.extend([0xF40C0205, 0xFA000000]);
    w.extend(vop3(0x346, 2, sgpr(5), 131, vgpr(0)));
    w.extend(vop3(0x346, 3, sgpr(6), 131, vgpr(1)));
    // Cell t=8*x+y, pair values t+1 and 2*(t+1).
    w.extend(vop3(0x346, 4, vgpr(0), 131, vgpr(1)));
    w.push(vop2(0x1A, 5, 131, 4)); // byte offset 8*t
    w.push(vop2(0x25, 6, 129, 4));
    w.push(vop2(0x1A, 7, 129, 6));
    w.extend([0xD8380100, 0x00070605]); // write2 v5, v6, v7 offsets 0/1
    w.push(vop2(0x25, 6, 192, 6)); // +64
    w.push(vop2(0x1A, 7, 129, 6));
    w.extend([0xD8388180, 0x00070605]); // t+64
    w.extend([0xBE90047E, 0x7D280084]); // save EXEC, cmpx 4 > v0
    let skip = w.len(); w.push(0);
    w.push(vop2(0x25, 4, 192, 4));
    w.push(vop2(0x1A, 5, 131, 4));
    w.push(vop2(0x25, 6, 192, 6));
    w.push(vop2(0x1A, 7, 129, 6));
    w.extend([0xD8388180, 0x00070605]); // t+128 for x < 4
    let join = w.len();
    w.push(0xBEFE0410); // restore full EXEC
    w.push(vop2(0x25, 8, 134, 0)); // center x+6
    // Read center and initialize the two independent accumulators.
    w.extend(vop3(0x346, 9, vgpr(8), 131, vgpr(1)));
    w.push(vop2(0x1A, 9, 131, 9));
    w.extend([0xD8DC0100, 0x14000009]); // read2 v[20:21], v9
    w.push(s_mov_b32(0, 129));
    let loop_head = w.len();
    // Each scalar iteration reads center ±(s0, s0+1, s0+2).
    for delta in 0..3u8 {
        w.push(sop2(2, 1, 0, 128 + delta)); // s_add_i32 s1,s0,delta
        w.push(sop2(3, 7, 128, 1)); // s_sub_i32 s7,0,s1
        for offset in [1u8, 7] {
            w.push(vop2(0x25, 9, sgpr(offset), 8));
            w.extend(vop3(0x346, 9, vgpr(9), 131, vgpr(1)));
            w.push(vop2(0x1A, 9, 131, 9));
            w.extend([0xD8DC0100, 0x12000009]); // read2 v[18:19],v9
            w.push(vop2(0x25, 20, vgpr(18), 20));
            w.push(vop2(0x25, 21, vgpr(19), 21));
        }
    }
    w.extend([sop2(2, 0, 0, 131), 0xBF068700]); // s0+=3; s0==7
    let again = w.len(); w.push(0);
    w.extend([vop1(6, 20, vgpr(20)), vop1(6, 21, vgpr(21)),
        vop1(1, 22, 128), vop1(1, 23, 128)]); // float results, zero sentinels
    w.extend([0xF0201F08, 0x00021402]); // image_store v[20:23],v[2:3],s[8:15]
    w.push(S_ENDPGM);
    w[skip] = 0xBF880000 | ((join as isize - skip as isize - 1) as i16 as u16 as u32);
    w[again] = 0xBF840000 | ((loop_head as isize - again as isize - 1) as i16 as u16 as u32);
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: true, vgprs: 32,
        layout: vec![
            RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
        ],
        used: vec![UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0,
            lower_bound: 0, stride: 0, dwords: 8, user_data: 4, slot: 1 }],
        io: StageIo::Cs(ComputeIo { rsrc2: (5 << 1) | (3 << 15) | (1 << 11) | (1 << 7) | (1 << 8), threads: [8,8,1] }),
        code: w.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let shader = lower_native_cs(&native, false);
    let c = crate::dxbc::Container::parse(&shader.dxil).unwrap();
    let psv = c.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()), u32::MAX);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root,pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: {e}"));
        let ud = vec![0u32; shader.user_data_dwords() as usize];
        let result = host.dispatch(&root,&pso,&Dispatch {
            roots: vec![(0,Root::Constants(ud))], views: &[View::TextureUav(16,16)],
            samplers: &[], groups: [2,2,1],
        }).unwrap_or_else(|e| panic!("{label}: {e}"));
        for y in 0..16 { for x in 0..16 {
            let sum = (0..13).map(|k| 8 * ((x % 8) + k) + y % 8 + 1).sum::<usize>();
            let at = 4*(y*16+x);
            assert_eq!(&result[0][at..at+4], &[(sum as f32).to_bits(), (2.0*sum as f32).to_bits(),0,0], "{label} {x},{y}");
        } }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The f8 count/allocate/publish protocol with one 64-lane native wave,
/// including a path that reaches the existing iteration cap. A raw UAV
/// substitutes for the original structured counter and particle records.
#[test]
fn certified_lds_allocation_numerical_and_cap() {
    fn branch(words: &mut [u32], at: usize, target: usize, opcode: u32) {
        words[at] = opcode | ((target as isize - at as isize - 1) as i16 as u16 as u32);
    }
    let mut w = vec![
        s_mov_b32(5,128),
        s_load_dwordx4(8,4)[0], s_load_dwordx4(8,4)[1],
        vop1(1,1,vgpr(0)),                  // keep local thread id
        0xD4C20010, 0x00020080,             // v_cmp_eq_u32 s[16:17], 0, v0
        0xBE922410,                         // s_and_saveexec_b64 s[18:19], s[16:17]
        vop1(1,0,128),
        0xD8380100, 0x00000000,             // ds_write2_b32 v0,v0,v0 offset1:1
        0xBEFE0412,                         // restore EXEC
        0xBE94047E,                         // s_mov_b64 s[20:21], exec
        vop1(1,3,193),                     // rank = -1
        0x7DA80200,                         // v_cmpx_gt_u32 exec, s0, v1
    ];
    let empty = w.len(); w.push(0);
    w.extend([
        vop1(1,3,129), vop1(1,2,128),
        0xD8800004, 0x03000302,             // ds_add_rtn_u32 v3,v2,v3 offset:4
        0x7DA40201,                         // v_cmpx_eq_u32 exec,s1,v1
    ]);
    let nocap = w.len(); w.push(0);
    w.push(s_mov_b32(22,128));
    let header = w.len();
    w.push(sop2(0,22,22,129));              // s_add_u32 s22,s22,1
    w.push(0xBF060216);                     // s_cmp_eq_u32 s22,s2
    let finished = w.len(); w.push(0);
    let again = w.len(); w.push(0);
    let gate = w.len();
    w.extend([0xBEFE0414, 0xBE922410]);     // restore full EXEC, select lane zero
    let skip = w.len(); w.push(0);
    w.extend([
        vop1(1,2,128), vop1(1,4,128),
        0xD8D80004, 0x05000002,             // ds_read_b32 v5,v2 offset:4
        0xE0C85000, 0x80020504,             // buffer_atomic_add v5,v4,s[8:11],0 offen glc
        0xD8340000, 0x00000502,             // ds_write_b32 v2,v5
    ]);
    let publish = w.len();
    w.extend([0xBEFE0412, 0x7D2206C1]);     // restore EXEC, rank >= 0
    let skip_store = w.len(); w.push(0);
    w.extend([
        vop1(1,2,128),
        0xD8D80000, 0x02000002,             // ds_read_b32 v2,v2
        vop2(0x25,3,vgpr(2),3),            // base + rank
        vop2(0x25,3,129,3),                // +1: distinguish untouched zero
        vop2(0x25,4,129,1),                // address = (thread id + 1)*4
        vop2(0x1A,4,130,4),
        buffer_store_dword_offen(3,4,8)[0],
        buffer_store_dword_offen(3,4,8)[1],
    ]);
    let end = w.len(); w.push(S_ENDPGM);
    branch(&mut w, empty, gate, 0xBF880000);
    branch(&mut w, nocap, gate, 0xBF880000);
    branch(&mut w, finished, gate, 0xBF850000);
    branch(&mut w, again, header, 0xBF820000);
    branch(&mut w, skip, publish, 0xBF880000);
    branch(&mut w, skip_store, end, 0xBF880000);
    for conditional in [false,true] {
    let mut words = w.clone();
    if conditional {
        branch(&mut words, finished, header, 0xBF840000);
        branch(&mut words, again, gate, 0xBF820000);
    }
    let shader = lower::with_allocation_test_limit(2, || build_cs_config(&words,64,true,512,true));
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root,pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: {e}"));
        for (count, loop_lane, bound, abort) in [
            (0,64,4,false),(1,64,4,false),(31,64,4,false),(32,64,4,false),
            (33,64,4,false),(64,64,4,false),(64,31,1,false),(64,31,4,true),
        ] {
            let inputs = [count, loop_lane, bound, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root,u.slot) {
                    (RootType::Constants,0) => inputs[u.dword as usize],
                    (RootType::Table,1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host.dispatch(&root,&pso,&Dispatch {
                roots: vec![(0,Root::Constants(ud))], views: &[View::RawUav(65)],
                samplers: &[], groups: [1,1,1],
            }).unwrap_or_else(|e| panic!("{label}: {e}"));
            if abort {
                assert!(out[0].iter().all(|&x| x == 0), "{label}: partial allocation on cap");
            } else {
                assert_eq!(out[0][0], count, "{label}: count");
                let mut ranks = out[0][1..=count as usize].to_vec();
                ranks.sort_unstable();
                assert_eq!(ranks, (1..=count).collect::<Vec<_>>(), "{label}: ranks");
                assert!(out[0][count as usize+1..].iter().all(|&x| x == 0), "{label}: nonparticipant wrote");
            }
        }
        ran += 1;
        eprintln!("{label}: allocation ranks and cap passed (conditional={conditional})");
    }
    assert!(ran > 0);
    }
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The f16 forms share the f32 forms' helper (operands widened first): -0
/// still orders below +0.
#[test]
fn f16_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f16", 0x139, pz),
        ("v_min_f16", 0x13A, nz),
        ("v_max3_f16", 0x354, pz),
        ("v_min3_f16", 0x351, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(
                got & 0xFFFF,
                want,
                "{backend}: {name} {inp:x?} = {got:#010x}"
            );
        });
    }
}

/// Integer three-way minimum, maximum and median, where signed and unsigned
/// order the same bits differently.
#[test]
fn integer_min3_max3_med3_match_rdna2() {
    let inputs: Vec<[u32; 4]> = [
        [5, 3, 9],
        [7, 7, 2],
        [0xFFFF_FFFF, 1, 0x8000_0000],
        [0x7FFF_FFFF, 0x8000_0000, 0],
        [0, 0xFFFF_FFFE, 0xFFFF_FFFF],
        [0x8000_0000, 0x8000_0000, 0x7FFF_FFFF],
    ]
    .iter()
    .map(|&[a, b, c]| [a, b, c, 0])
    .collect();

    // Sorted operands, then the one the instruction picks.
    fn pick(inp: [u32; 4], signed: bool, which: usize) -> u32 {
        let mut v = [inp[0], inp[1], inp[2]];
        if signed {
            v.sort_by_key(|&x| x as i32);
        } else {
            v.sort();
        }
        v[which]
    }
    let ops = [
        ("v_min3_i32", 0x152, true, 0),
        ("v_min3_u32", 0x153, false, 0),
        ("v_max3_i32", 0x155, true, 2),
        ("v_max3_u32", 0x156, false, 2),
        ("v_med3_i32", 0x158, true, 1),
        ("v_med3_u32", 0x159, false, 1),
    ];
    for (name, op, signed, which) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            let want = pick(inp, signed, which);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// `v_cmp_f_*` is false and `v_cmp_tru_*` true whatever the operands, NaNs
/// included (RDNA2 ISA). As `fcmp false`/`fcmp true` WARP removed its device
/// at the first dispatch.
#[test]
fn float_compares_that_are_always_false_or_true() {
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;
    let f32_inputs = [
        [f(1.0), f(2.0), 0, 0],
        [f(1.0), f(1.0), 0, 0],
        [nan, f(1.0), 0, 0],
        [nan, nan, 0, 0],
    ];
    let f16_inputs = [
        [0x3C00, 0x4000, 0, 0],
        [0x3C00, 0x3C00, 0, 0],
        [0x7E00, 0x3C00, 0, 0],
        [0x7E00, 0x7E00, 0, 0],
    ];
    // v_cmp_*_e64 s20, v1, v2; v_cndmask_b32_e64 v0, 0, 1, s20
    for (op, what, inputs, want) in [
        (0x000u16, "v_cmp_f_f32", f32_inputs, 0),
        (0x00f, "v_cmp_tru_f32", f32_inputs, 1),
        (0x0c8, "v_cmp_f_f16", f16_inputs, 0),
        (0x0ef, "v_cmp_tru_f16", f16_inputs, 1),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &inputs, |backend, _, inp, got| {
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27); v_mov_b32 v0, s6. Its width
    // is 7-bit too: 32..127 keeps every bit from the offset up, unextended.
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.extend([
        ([0x8000_0000, 4, 32, 0], 0x0800_0000),
        ([0xFFFF_FFFF, 0, 127, 0], 0xFFFF_FFFF),
        ([0xF000_0000, 28, 64, 0], 0x0000_000F),
    ]);
    let su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    let build = |abs0: bool, neg0: bool| {
        let [mut w0, mut w1] = vop3(0x088, 6, vgpr(1), vgpr(2), 0);
        if abs0 {
            w0 |= 1 << 8; // abs[0] = 1
        }
        if neg0 {
            w1 |= 1 << 29; // neg[0] = 1
        }
        let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        build_valu_cs(&[w0, w1, sel[0], sel[1]])
    };

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x40_0000 clear)
        (0xFF80_0001, 0), // -sNaN
        (0x7FC0_0000, 1), // +qNaN (quiet bit set)
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let mut plain_cases = Vec::new();
    for &(bits, cls) in classes {
        plain_cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        plain_cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let cmp_plain = build(false, false);
    let plain_in: Vec<[u32; 4]> = plain_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_plain, &plain_in, |backend, i, inp, got| {
        let want = plain_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // RDNA2 input modifiers on v_cmp_class_f32 manipulate the sign bit bitwise:
    // they must not quiet sNaNs or flush denormals to zero.
    struct ModCases {
        label: &'static str,
        abs: bool,
        neg: bool,
        items: &'static [(u32, u32)],
    }
    let mod_sets = [
        ModCases {
            label: "neg",
            abs: false,
            neg: true,
            items: &[
                (0x3F80_0000, 3), // -(+1.0) -> -normal
                (0x0000_0000, 5), // -(+0.0) -> -0.0
                (0x8000_0000, 6), // -(-0.0) -> +0.0
                (0x0000_0001, 4), // -(+denormal) -> -denormal (not flushed to -0.0)
                (0x8000_0001, 7), // -(-denormal) -> +denormal (not flushed to +0.0)
                (0x7F80_0001, 0), // -(+sNaN) -> sNaN (not quieted to qNaN)
                (0xFF80_0001, 0), // -(-sNaN) -> sNaN (not quieted to qNaN)
            ],
        },
        ModCases {
            label: "abs",
            abs: true,
            neg: false,
            items: &[
                (0xBF80_0000, 8), // |-1.0| -> +normal
                (0x8000_0000, 6), // |-0.0| -> +0.0
                (0xFF80_0000, 9), // |-inf| -> +inf
                (0x8000_0001, 7), // |-denormal| -> +denormal
                (0xFF80_0001, 0), // |-sNaN| -> sNaN
            ],
        },
        ModCases {
            label: "abs+neg",
            abs: true,
            neg: true,
            items: &[
                (0x3F80_0000, 3), // -|+1.0| -> -normal
                (0xBF80_0000, 3), // -|-1.0| -> -normal
                (0x0000_0001, 4), // -|+denormal| -> -denormal
                (0x8000_0001, 4), // -|-denormal| -> -denormal
                (0x7F80_0001, 0), // -|+sNaN| -> sNaN
            ],
        },
    ];
    for set in mod_sets {
        let shader = build(set.abs, set.neg);
        let mut cases = Vec::new();
        for &(bits, cls) in set.items {
            cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
            cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
        }
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: v_cmp_class_f32[{}]({:#010x}, {:#05x}) = {got}, want {want}",
                set.label, inp[0], inp[1],
            );
        });
    }
}

#[test]
fn sdwa_16bit_dst_unused_sext_sign_extends_word0() {
    // SDWA dst_sel: 4 = WORD_0, 5 = WORD_1.
    // SDWA dst_unused: 0 = UNUSED_PAD, 1 = UNUSED_SEXT, 2 = UNUSED_PRESERVE.
    // Pre-seed v0 from v4 (0xABCD_1234) so UNUSED_SEXT/UNUSED_PAD/UNUSED_PRESERVE
    // are distinguishable from each other.
    let seed = 0xABCD_1234u32;

    // v_cvt_i16_f16_sdwa v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop1_sdwa(0x053, 0, 1, 4, 1, 4));
    let cvt_sext = build_valu_cs(&body);
    let cvt_cases: &[(u32, u32)] = &[
        (0xBC00, 0xFFFF_FFFF), // -1.0 -> -1i16 sign-extended to -1i32
        (0xF800, 0xFFFF_8000), // -32768.0 -> -32768i16 sign-extended
        (0x3C00, 0x0000_0001), // +1.0 -> +1i16 zero-extended (old v0 upper bits cleared)
        (0x7BFF, 0x0000_7FFF), // +65504.0 -> saturated +32767i16
        (0x0000, 0x0000_0000), // 0.0 -> 0
    ];
    let cvt_in: Vec<[u32; 4]> = cvt_cases.iter().map(|&(x, _)| [x, 0, 0, seed]).collect();
    run_cases(&cvt_sext, &cvt_in, |backend, i, inp, got| {
        let want = cvt_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cvt_i16_f16_sdwa({:#06x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // v_cvt_norm_i16_f16_sdwa v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop1_sdwa(0x063, 0, 1, 4, 1, 5));
    let norm_sext = build_valu_cs(&body);
    let norm_cases: &[(u32, u32)] = &[
        (0xBC00_3C00, 0xFFFF_8001), // high half -1.0 -> -32767i16 sign-extended
        (0x3C00_BC00, 0x0000_7FFF), // high half +1.0 -> +32767i16
    ];
    let norm_in: Vec<[u32; 4]> = norm_cases.iter().map(|&(x, _)| [x, 0, 0, seed]).collect();
    run_cases(&norm_sext, &norm_in, |backend, i, inp, got| {
        let want = norm_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cvt_norm_i16_f16_sdwa({:#010x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // v_add_f16_sdwa v0, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:WORD_0
    let mut body = vec![vop1(0x001, 0, vgpr(4))];
    body.extend_from_slice(&vop2_sdwa(0x32, 0, 1, 2, (4, 1), [4, 4]));
    let add_sext = build_valu_cs(&body);
    let add_cases: &[(u32, u32, u32)] = &[
        (0xBC00, 0xC000, 0xFFFF_C200), // -1.0 + -2.0 = -3.0 (0xC200, bit 15 set -> sign-extended)
        (0x3C00, 0x4000, 0x0000_4200), // +1.0 + +2.0 = +3.0 (0x4200, bit 15 clear)
    ];
    let add_in: Vec<[u32; 4]> = add_cases.iter().map(|&(a, b, _)| [a, b, 0, seed]).collect();
    run_cases(&add_sext, &add_in, |backend, i, inp, got| {
        let want = add_cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_add_f16_sdwa({:#06x}, {:#06x}, WORD_0, UNUSED_SEXT) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    // Regression check for the other dst_sel / dst_unused modes on v_cvt_i16_f16_sdwa(-1.0):
    // WORD_0 + UNUSED_PAD -> 0x0000_FFFF, WORD_0 + UNUSED_PRESERVE -> 0xABCD_FFFF,
    // WORD_1 + UNUSED_PAD / UNUSED_SEXT -> 0xFFFF_0000 (bits below WORD_1 pad with 0),
    // WORD_1 + UNUSED_PRESERVE -> 0xFFFF_1234, DWORD -> 0x0000_FFFF.
    let modes: &[(u8, u8, u32)] = &[
        (4, 0, 0x0000_FFFF),
        (4, 2, 0xABCD_FFFF),
        (5, 0, 0xFFFF_0000),
        (5, 1, 0xFFFF_0000),
        (5, 2, 0xFFFF_1234),
        (6, 0, 0x0000_FFFF),
    ];
    for &(dst_sel, dst_unused, want) in modes {
        let mut body = vec![vop1(0x001, 0, vgpr(4))];
        body.extend_from_slice(&vop1_sdwa(0x053, 0, 1, dst_sel, dst_unused, 4));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &[[0xBC00, 0, 0, seed]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_i16_f16_sdwa(-1.0, dst_sel={dst_sel}, dst_unused={dst_unused}) = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn bounded_m0_moves_do_not_read_descriptor_words() {
    let mut body = Vec::new();
    for k in 0..4u8 {
        body.push(vop1(0x001, 8+k, 138 + u16::from(k)));
        body.push(vop1(0x001, 12+k, sgpr(8+k)));
    }
    body.extend(vop3(0x159, 5, vgpr(1), 128, 131));
    body.push(vop1(0x002, 12, vgpr(5)));
    body.push(s_mov_b32(124, 12));
    body.push(vop1(0x042, 8, vgpr(3)));
    body.push(0x1B << 25 | 6 << 17 | 2 << 9 | 131);
    body.push(vop1(0x002, 12, vgpr(6)));
    body.push(s_mov_b32(124, 12));
    body.push(vop1(0x043, 0, vgpr(8)));
    let shader = build_valu_cs_vgprs(&body, 16);
    let cases = [[0,0,77,0], [5,3,77,0], [1,2,77,0], [u32::MAX,7,77,0], [2,1,77,0], [3,0,77,0]];
    run_cases(&shader, &cases, |backend, _, [a,b,c,_], got| {
        let (i,j) = (a.min(3), b & 3);
        assert_eq!(got, if i == j { c } else { 10+j }, "{backend}: {a:#x} {b:#x}");
    });
}

#[test]
fn m0_relative_swaps_and_moves_index_source_and_destination_apart() {
    // v1..v4 = 10, 20, 30, 40. v_movrelsd_2_b32 and v_swaprel_b32 offset
    // their source by M0[9:0] and their destination by M0[25:16]; other M0
    // bits are ignored. Each case reads one VGPR back into v0.
    const SWAPREL: u16 = 0x068;
    const MOVRELSD_2: u16 = 0x048;
    let cases: &[(u16, u8, u32, u8, u32)] = &[
        // (opcode, vdst, M0, register read back, expected)
        // v[1 + 2] <-> v[1 + 0]
        (SWAPREL, 1, 2 << 16, 1, 30),
        (SWAPREL, 1, 2 << 16, 3, 10),
        // v[1 + 0] <-> v[1 + 3]
        (SWAPREL, 1, 3, 4, 10),
        (SWAPREL, 1, 3, 1, 40),
        // Both name v2: it keeps its value.
        (SWAPREL, 1, 1 << 16 | 1, 2, 20),
        // v[5 + 1] = v[1 + 2]
        (MOVRELSD_2, 5, 1 << 16 | 2, 6, 30),
        // The same with every ignored M0 bit set.
        (MOVRELSD_2, 5, 0xFC00_FC00 | 1 << 16 | 2, 6, 30),
    ];
    for &(op, vdst, m0, back, want) in cases {
        let shader = build_valu_cs(&[
            s_mov_b32(124, 0xFF), // s_mov_b32 m0, literal
            m0,
            vop1(op, vdst, vgpr(1)),
            vop1(0x001, 0, vgpr(back)), // v_mov_b32 v0, v[back]
        ]);
        run_cases(&shader, &[[10, 20, 30, 40]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: op {op:#05x} v{vdst}, v1 with M0 {m0:#010x}, v{back} = {got}, want {want}",
            );
        });
    }
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

/// Runs VOP1 `op` on each input and compares the result dword.
fn check_cvt(name: &str, op: u16, cases: &[(u32, u32)]) {
    let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [c.0, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got,
            want,
            "{backend}: {name}({:#010x} = {:e}) = {got:#010x}, want {want:#010x}",
            inp[0],
            f32::from_bits(inp[0]),
        );
    });
}

#[test]
fn v_cvt_rpi_i32_f32_rounds_to_nearest() {
    // floor(x + 0.5) of the exact sum: an f32 sum rounds to even first, so
    // it is 1 too high just below 0.5 and for odd integers from 2^23 up.
    check_cvt(
        "v_cvt_rpi_i32_f32",
        0x00C,
        &[
            (0x0000_0000, 0),
            (0x3EFF_FFFF, 0), // 0.5 - 2^-25
            (0x3F00_0000, 1), // 0.5: ties go up
            (0xBF00_0000, 0), // -0.5
            (0xBF00_0001, 0xFFFF_FFFF),
            (0x3FC0_0000, 2),           // 1.5
            (0x4020_0000, 3),           // 2.5
            (0xBFC0_0000, 0xFFFF_FFFF), // -1.5
            (0xC020_0000, 0xFFFF_FFFE), // -2.5
            (0x0000_0001, 0),           // denormals, flushed or not
            (0x8000_0001, 0),
            (0x4A80_0001, 0x0040_0001), // 2^22 + 0.5
            (0x4B00_0001, 0x0080_0001), // 2^23 + 1
            (0x4B7F_FFFF, 0x00FF_FFFF), // 2^24 - 1
            (0xCB00_0001, 0xFF7F_FFFF), // -(2^23 + 1)
            (0xCB7F_FFFF, 0xFF00_0001),
            (0x4EFF_FFFF, 0x7FFF_FF80),
            (0x4F00_0000, 0x7FFF_FFFF), // 2^31 and up saturate
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0000, 0x8000_0000),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_i32_f32_rounds_down() {
    check_cvt(
        "v_cvt_flr_i32_f32",
        0x00D,
        &[
            (0x0000_0000, 0),
            (0x8000_0000, 0),
            (0x3FC0_0000, 1),           // 1.5
            (0xBFC0_0000, 0xFFFF_FFFE), // -1.5
            (0x3F7F_FFFF, 0),
            (0xBF80_0000, 0xFFFF_FFFF), // -1
            (0xB300_0000, 0xFFFF_FFFF), // -2^-25
            (0x4B7F_FFFF, 0x00FF_FFFF),
            (0x4F00_0000, 0x7FFF_FFFF),
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_and_rpi_i32_f32_saturate_nan_by_sign() {
    // Unlike v_cvt_i32_f32 (NaN gives 0), these saturate whatever has an
    // exponent above 30, a NaN too: to the largest or the smallest integer
    // by its sign.
    let cases = [
        (0x7FC0_0000, 0x7FFF_FFFF),
        (0x7F80_0001, 0x7FFF_FFFF),
        (0x7FFF_FFFF, 0x7FFF_FFFF),
        (0xFFC0_0000, 0x8000_0000),
        (0xFF80_0001, 0x8000_0000),
        (0xFFFF_FFFF, 0x8000_0000),
    ];
    check_cvt("v_cvt_rpi_i32_f32", 0x00C, &cases);
    check_cvt("v_cvt_flr_i32_f32", 0x00D, &cases);
    // v_cvt_i32_f32 keeps giving 0.
    check_cvt(
        "v_cvt_i32_f32",
        0x008,
        &[(0x7FC0_0000, 0), (0xFFC0_0000, 0), (0x7F80_0001, 0)],
    );
}

/// One wave of 32 lanes holding 1..=32 (the input of the ISA's examples)
/// through `ds_swizzle_b32` with `offset`: what each lane reads.
fn swizzle_32(offset: u16) -> Vec<Vec<u32>> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x25, 1, 129, 0)); // v_add_nc_u32 v1, 1, v0
    words.extend_from_slice(&ds_swizzle_b32(2, 1, offset));
    words.push(vop2(0x1A, 3, 130, 0)); // v_lshlrev_b32 v3, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(2, 3, 8));
    words.push(S_ENDPGM);
    let shader = build_cs(&words, 32);

    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // A host without 32-lane waves cannot run it.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push(uavs[0].clone());
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device with 32-lane waves; skipped");
    }
    out
}

/// Exercise the logical wave64 with two-dimensional native thread IDs,
/// swizzles in both 32-lane halves, fixed reads across halves, and reuse of
/// an overwritten VGPR (distinct SSA snapshot). Integer payloads make every
/// bit of lane selection observable without float rounding tolerances.
#[test]
fn logical_lane_snapshot_numerical() {
    let mut words = vec![s_mov_b32(5, 128)];
    words.extend(s_load_dwordx4(8,4));
    words.extend(vop3(0x346,1,vgpr(1),131,vgpr(0))); // v1 = y*8+x
    words.push(vop2(0x25,2,129,1));                   // v2 = flat+1
    words.extend(ds_swizzle_b32(3,2,0x401f));         // xor16
    words.extend(ds_swizzle_b32(6,2,0x101f));         // xor4; same snapshot
    words.extend(vop3(0x360,6,vgpr(2),160,0));         // s6 = lane32
    words.extend(vop3(0x360,7,vgpr(2),128,0));         // s7 = lane0
    words.push(vop2(0x25,2,129,2));                   // v2 = flat+2 (new SSA)
    words.extend(vop3(0x360,12,vgpr(2),164,0));        // s12 = lane36
    words.push(vop2(0x1A,4,133,1));                   // byte address flat*32
    for (offset, reg, scalar) in [(0,3,false),(4,6,false),(8,6,true),(12,7,true),(16,12,true)] {
        let data = if scalar { words.push(vop1(1,5,sgpr(reg))); 5 } else { reg };
        let mut store = buffer_store_dword_offen(data,4,8);
        store[0] |= offset;
        words.extend(store);
    }
    words.push(S_ENDPGM);
    let shader = build_cs_group(&words,[8,8,1],true,0,false);
    let container = crate::dxbc::Container::parse(&shader.dxil).unwrap();
    let flags = u64::from_le_bytes(container.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap());
    assert_eq!(flags & 0x4000,0);
    let psv = container.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()),0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()),u32::MAX);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware",false),("warp",true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root,pso) = host.compute_pipeline(&shader).unwrap_or_else(|e| panic!("{label}: {e}"));
        let ud = vec![0u32;shader.user_data_dwords().max(1) as usize];
        let out = host.dispatch(&root,&pso,&Dispatch { roots:vec![(0,Root::Constants(ud))],
            views:&[View::RawUav(64*8)],samplers:&[],groups:[1,1,1] }).unwrap_or_else(|e| panic!("{label}: {e}"));
        for (lane, record) in out[0].as_chunks::<8>().0.iter().enumerate() {
            assert_eq!(record, &[(lane as u32^16)+1,(lane as u32^4)+1,33,1,38,0,0,0],
                "{label}: lane {lane}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn logical_lane_hierarchical_reduction_numerical() {
    let mut w = vec![s_mov_b32(5,128)];
    w.extend(s_load_dwordx4(8,4));
    w.extend(vop3(0x346,1,vgpr(1),131,vgpr(0))); // flat ID
    // Strictly decreasing input, so the minimum of a neighborhood may
    // come from a different host warp. All inputs and arithmetic exact.
    w.push(vop2(0x26,3,192,1)); // 64 - flat
    w.push(vop2(0x1A,4,132,1)); // byte address = 16*flat
    w.extend(ds_swizzle_b32(5,3,0x201f)); // xor 8
    w.push(vop2(0x13,3,vgpr(5),3)); // min
    w.extend(ds_swizzle_b32(5,3,0x041f)); // xor 1
    w.push(vop2(0x13,3,vgpr(5),3));
    w.extend(vop3(0x365,2,193,128,0)); // mbcnt lo(-1,0)
    w.extend(vop3(0x366,2,193,vgpr(2),0)); // mbcnt hi(-1,lo)
    w.push(vop2(0x1b,6,137,2)); // lane & 9
    w.extend([0xBE90047E,0x7DA40C80]); // save EXEC, cmpx_eq 0,v6
    let skip = w.len(); w.push(0);
    w.extend(buffer_store_dword_offen(3,4,8));
    let join = w.len(); w.push(0xBEFE0410); // restore full EXEC
    w[skip] = 0xBF880000 | ((join as isize-skip as isize-1) as i16 as u16 as u32);
    for offset in [0x401f,0x081f] { // xor 16,2
        w.extend(ds_swizzle_b32(5,3,offset));
        w.push(vop2(0x13,3,vgpr(5),3));
    }
    w.push(vop2(0x1b,6,155,2)); // lane & 27
    w.extend([0xBE90047E,0x7DA40C80]);
    let skip = w.len(); w.push(0);
    let mut store = buffer_store_dword_offen(3,4,8); store[0] |= 4; w.extend(store);
    let join = w.len(); w.push(0xBEFE0410);
    w[skip] = 0xBF880000 | ((join as isize-skip as isize-1) as i16 as u16 as u32);
    w.extend(ds_swizzle_b32(5,3,0x101f)); // xor 4: full 32-lane half
    w.push(vop2(0x13,3,vgpr(5),3));
    w.extend(vop3(0x360,12,vgpr(3),160,0)); // scalar lane 32
    // For every lane, test the half reduction and exact lane number.
    let mut store = buffer_store_dword_offen(3,4,8); store[0] |= 8; w.extend(store);
    w.extend([0xBE90047E,0x7DA40480]); // only lane 0 commits full reduction
    let skip = w.len(); w.push(0);
    w.push(vop2(0x13,3,sgpr(12),3));
    let mut store = buffer_store_dword_offen(3,4,8); store[0] |= 12; w.extend(store);
    let end = w.len(); w.push(S_ENDPGM);
    w[skip] = 0xBF880000 | ((end as isize-skip as isize-1) as i16 as u16 as u32);
    let shader = build_cs_group(&w,[8,8,1],true,0,false);
    let c = crate::dxbc::Container::parse(&shader.dxil).unwrap();
    assert_eq!(u64::from_le_bytes(c.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap()) & 0x4000,0);
    let psv = c.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()),0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()),u32::MAX);
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e|e.into_inner());
    let mut ran = 0;
    for (label,warp) in [("hardware",false),("warp",true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root,pso) = host.compute_pipeline(&shader).unwrap_or_else(|e|panic!("{label}: {e}"));
        let ud = vec![0;shader.user_data_dwords().max(1) as usize];
        let out = host.dispatch(&root,&pso,&Dispatch { roots:vec![(0,Root::Constants(ud))],views:&[View::RawUav(64*4)],samplers:&[],groups:[1,1,1] }).unwrap_or_else(|e|panic!("{label}: {e}"));
        for (lane,record) in out[0].as_chunks::<4>().0.iter().enumerate() {
            let l = lane as u32;
            assert_eq!(record,&[if l&9==0 {64-(l|9)} else {0},if l&27==0 {64-(l|27)} else {0},if l<32 {33} else {1},if l==0 {1} else {0}],"{label}: lane {l}");
        }
        ran += 1;
    }
    assert!(ran>0);
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn ds_swizzle_basic_modes() {
    // offset[15]: lanes of each quad read lanes 3, 2, 1, 0 of it.
    let reversed: Vec<u32> = (0..32).map(|i| (i ^ 3) + 1).collect();
    // and 0x1f, or 0, xor 0x10: the halves swap.
    let swapped: Vec<u32> = (0..32).map(|i| (i ^ 16) + 1).collect();
    for (offset, want) in [(0x801B, reversed), (0x401F, swapped)] {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Offsets from 0xE000: the FFT orders, as the ISA lists them for inputs
/// 1..=0x20 (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_fft_mode() {
    let cases: [(u16, [u32; 32]); 3] = [
        (
            0xE000,
            [
                0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17,
                0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c,
                0x08, 0x18, 0x10, 0x20,
            ],
        ),
        (
            0xE010,
            [
                0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c,
                0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e,
                0x14, 0x1c, 0x18, 0x20,
            ],
        ),
        // 0x1f: no swizzle.
        (0xE01F, std::array::from_fn(|i| i as u32 + 1)),
    ];
    for (offset, want) in cases {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
    // The other bits of an FFT offset do not matter.
    for got in swizzle_32(0xFFF0) {
        assert_eq!(got, cases[1].1, "offset 0xfff0");
    }
}

/// Offsets from 0xC000 to 0xDFFF: the wave's 32 lanes rotate left by
/// offset[9:5], or right with offset[10] (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_rotate_mode() {
    let rotated = |by: u32| -> Vec<u32> { (0..32).map(|i| (i + by) % 32 + 1).collect() };
    for (offset, by) in [
        (0xC000 | 1 << 5, 1),            // the ISA's example: {2, 3, ... 0x20, 1}
        (0xC000 | 1 << 10 | 1 << 5, 31), // and {0x20, 1, ... 0x1f}
        (0xC000, 0),
        (0xC000 | 5 << 5, 5),
        (0xC000 | 1 << 10 | 5 << 5, 27),
        (0xD800 | 31 << 5, 31), // bits 11 and 12 are not read
    ] {
        for got in swizzle_32(offset) {
            assert_eq!(got, rotated(by), "offset {offset:#06x}");
        }
    }
}

/// Rotations with a mask (offset[4:0]): the mask's bits of the lane index
/// stay, the others rotate, as the ISA's pseudocode has it. Its examples for
/// mask 1 list other lanes; the lanes here are the ones measured on RDNA2
/// hardware (gfx1036, by Lyrnc in Marvisak/VIctory#606, who gives the first
/// and last entries of each list and reports the pseudocode matching all 2048
/// offsets 0xC000-0xC7FF, and bits 11 and 12 being ignored).
#[test]
fn ds_swizzle_masked_rotations() {
    // The lane each lane reads.
    let cases: [(u16, [u32; 32]); 4] = [
        // Left by 1, mask 1: 0, 3, 2, 5, 4, 7, 6, 9, 8, ..., 28, 31, 30, 1.
        (
            0xC021,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
        // Right by 1, mask 1: 30, 1, 0, 3, 2, 5, 4, ...
        (
            0xC421,
            [
                30, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                23, 22, 25, 24, 27, 26, 29, 28, 31,
            ],
        ),
        // Left by 1, mask 3: 0, 1, 2, 7, 4, 5, 6, 11, ...
        (
            0xC023,
            [
                0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22,
                27, 24, 25, 26, 31, 28, 29, 30, 3,
            ],
        ),
        // Bits 11 and 12 set: as 0xC021.
        (
            0xD821,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
    ];
    for (offset, lanes) in cases {
        // Lane i holds i + 1.
        let want: Vec<u32> = lanes.iter().map(|j| j + 1).collect();
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// A 16-bit snorm (`signed`) or unorm of `x`, as the RDNA2 ISA reference
/// gives `v_cvt_pknorm_*`: clamped, scaled, rounded to nearest even from the
/// exact product (exact in f64), NaN as 0.
fn norm16_ref(x: f32, signed: bool) -> u16 {
    if x.is_nan() {
        return 0;
    }
    let (lo, k) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    let n = (f64::from(x.clamp(lo, 1.0)) * k).round_ties_even() as i32;
    n as u16
}

/// Inputs in [0.5, 1) and near 0.001 whose f32 product with the scale lands
/// exactly on a tie although the exact product does not: a translation that
/// multiplies in f32 rounds them the wrong way.
fn f32_tie_traps(signed: bool) -> Vec<f32> {
    let k: f32 = if signed { 32767.0 } else { 65535.0 };
    let mut traps = Vec::new();
    for start in [0.5f32, 0.001] {
        let first = start.to_bits();
        for b in first..first + 4_000_000 {
            let x = f32::from_bits(b);
            let p = x * k;
            if p.fract() == 0.5 && f64::from(x) * f64::from(k) != f64::from(p) {
                traps.push(x);
                if traps.len() % 3 == 0 {
                    break;
                }
            }
        }
    }
    traps
}

fn pknorm_f32_inputs(signed: bool) -> Vec<f32> {
    let mut xs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        0.5,
        -0.5,
        0.25,
        1.0 / 3.0,
        -0.7,
        0.999_99,
        f32::MIN_POSITIVE,
        1e-6,
        -1e-6,
        f32::from_bits(1), // denormal
        1.0 / 65535.0,
        0.5 / 65535.0,
        1.5 / 32767.0,
        f32::from_bits(0x3F7F_FFFF), // just below 1
    ];
    xs.extend(f32_tie_traps(signed));
    xs
}

#[test]
fn v_cvt_pknorm_f32_rounds_the_exact_product() {
    for (op, signed, what) in [
        (0x368u16, true, "v_cvt_pknorm_i16_f32"),
        (0x369, false, "v_cvt_pknorm_u16_f32"),
    ] {
        // The traps must tell an f32 product apart from the exact one.
        let k: f32 = if signed { 32767.0 } else { 65535.0 };
        let f32_product = |x: f32| (x * k).round_ties_even() as i32 as u16;
        let traps = f32_tie_traps(signed);
        assert!(
            traps
                .iter()
                .any(|&x| f32_product(x) != norm16_ref(x, signed)),
            "{what}: no input separates an f32 product from the exact one"
        );
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), 0));
        let xs = pknorm_f32_inputs(signed);
        // Each input once in each half, paired with its neighbour.
        let cases: Vec<[u32; 4]> = (0..xs.len())
            .map(|i| [xs[i].to_bits(), xs[(i + 1) % xs.len()].to_bits(), 0, 0])
            .collect();
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let (a, b) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
            let want = u32::from(norm16_ref(a, signed)) | u32::from(norm16_ref(b, signed)) << 16;
            assert_eq!(
                got, want,
                "{backend}: {what}({a:e} [{:#010x}], {b:e} [{:#010x}]) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_cvt_pknorm_f16_rounds_the_exact_product() {
    let halves: Vec<u16> = vec![
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x7C00, 0xFC00, 0x7E00, 0xFE00,
        0x3800, // 0.5: a tie in both scales
        0xB800, 0x3555, 0x0001, // smallest denormal
        0x8001, 0x0400, // smallest normal
        0x3BFF, // just below 1
        0x1001, 0x2E66, 0xB266, 0x3A00, 0x3401,
        // Large denormals, which scale to 1 or 2: kept, not flushed (the
        // FP16 input-denormal mode RDNA2 measurements matched).
        0x03D6, 0x83D6, 0x823F, 0x03FF,
    ];
    // Measured on RDNA2 (gfx1036) with FP16 input denormals kept.
    assert_eq!(norm16_ref(f16_to_f32(0x83D6), true), 0xFFFE);
    assert_eq!(norm16_ref(f16_to_f32(0x823F), true), 0xFFFF);
    for (op, signed, what) in [
        (0x312u16, true, "v_cvt_pknorm_i16_f16"),
        (0x313, false, "v_cvt_pknorm_u16_f16"),
    ] {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), 0));
        let cases: Vec<[u32; 4]> = (0..halves.len())
            .map(|i| {
                let next = halves[(i + 1) % halves.len()];
                // Junk in the high halves: only the low halves are read.
                [
                    u32::from(halves[i]) | 0xABCD_0000,
                    u32::from(next) | 0x1234_0000,
                    0,
                    0,
                ]
            })
            .collect();
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let (a, b) = (inp[0] as u16, inp[1] as u16);
            let (fa, fb) = (f16_to_f32(a), f16_to_f32(b));
            let want = u32::from(norm16_ref(fa, signed)) | u32::from(norm16_ref(fb, signed)) << 16;
            assert_eq!(
                got, want,
                "{backend}: {what}({a:#06x}, {b:#06x}) = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

/// MUBUF `buffer_load_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_load_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A root SRV has no bounds on the host: a read past what the host bound
/// reads 0, as a console buffer view past its records does, instead of
/// memory after it (or an unmapped page, which some GPUs stall on).
#[test]
fn root_view_reads_stop_at_the_bound_size() {
    // s0..s3 root constants, s4 the UAV table, s[8:9] the root SRV's address.
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(12, 4));
    // The buffer descriptor built around the root address: all records.
    words.push(s_mov_b32(10, 0xC1));
    words.push(s_mov_b32(11, 0x80));
    words.push(vop1(0x001, 1, sgpr(0))); // v_mov_b32 v1, s0: the byte offset
    words.extend_from_slice(&buffer_load_dword_offen(0, 1, 8));
    words.extend_from_slice(&buffer_store_dword(0, 12, 0));
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let root = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, root, user_data, slot| UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            root(RootType::Constants, 4, 0, 0),
            root(RootType::Table, 1, 4, 1),
            root(RootType::Descriptor, 2, 8, 2),
        ],
        used: vec![
            used(ResourceType::Uav, RootType::Table, 4, 1),
            used(ResourceType::Srv, RootType::Descriptor, 8, 2),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 10 << 1, // 10 user SGPRs: s0..s9
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        static_table_cbvs: lowered.static_table_cbvs,
        heap_buffer_classes: lowered.heap_buffer_classes,
        heap_demands: lowered.heap_demands,
    };
    assert_eq!(shader.root_srvs, [2]);
    assert!(
        shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Descriptor && u.slot == 2 && u.index == 8),
        "the root view's size is user data: {:?}",
        shader.user_data
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    // 64 dwords bound, of which the host says 16 (64 bytes) are the view's.
    let data: Vec<u32> = (0..64).map(|i| 0x100 + i).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        for (offset, want) in [
            (0, 0x100),
            (60, 0x10F),
            (64, 0),
            (128, 0),
            (u32::MAX - 3, 0),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => offset,
                    RootType::Descriptor => 64,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: data.clone(),
                                },
                            ),
                        ],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(out[0][0], want, "{label}: offset {offset:#x}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
