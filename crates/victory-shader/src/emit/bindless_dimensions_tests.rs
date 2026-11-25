//! Regression tests using original title shaders kept outside the repository.
//! Set `VICTORY_PIXEL_DESCRIPTOR_DIR` to the directory holding either
//! `pixel-<hash>.cso` or `pixel-one-<hash>/pixel-<hash>.cso`.

use super::{translate, validator};

/// Original pixel shaders that previously refused direct barycentric data.
#[test]
fn original_pixel_barycentric_inputs_validate() {
    let Some(dir) = std::env::var_os("VICTORY_PIXEL_DESCRIPTOR_DIR") else { return; };
    let mut v = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for original fixture");
    for hash in ["2d0cace65bc05bb2", "98bd9f9ae5f9c7e3"] {
        let path = std::path::PathBuf::from(&dir).join(format!("pixel-{hash}.cso"));
        let bytes = std::fs::read(path).expect("external native pixel fixture");
        let shader = translate(&bytes).unwrap_or_else(|e| panic!("{hash}: {e}"));
        v.validate(&mut shader.dxil.clone()).unwrap_or_else(|e| panic!("{hash}: {e}"));
        eprintln!("{hash}: validated");
    }
}

#[test]
fn original_bindless_texture_dimensions_validate() {
    let Some(dir) = std::env::var_os("VICTORY_PIXEL_DESCRIPTOR_DIR") else {
        eprintln!("VICTORY_PIXEL_DESCRIPTOR_DIR not set; original shaders stay external");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required when shader fixtures are supplied");
    for hash in ["6b6db9f1a526daeb", "7d0f14ce44746d7e"] {
        let name = format!("pixel-{hash}.cso");
        let flat = dir.join(&name);
        let nested = dir.join(format!("pixel-one-{hash}")).join(&name);
        let path = if flat.is_file() { flat } else { nested };
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let shader = translate(&bytes).unwrap_or_else(|e| panic!("{hash}: {e}"));
        let mut dxil = shader.dxil.clone();
        validator
            .validate(&mut dxil)
            .unwrap_or_else(|e| panic!("{hash}: invalid DXIL: {e}"));
    }
}

/// Original compute program dynamically chooses a structured view from the
/// resource heap, then performs indexed reads and writes with its descriptor
/// byte stride. Keep the fixture outside this public source repository.
#[test]
fn original_dynamic_resource_heap_stride_validates() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else {
        eprintln!("VICTORY_COMPUTE_DESCRIPTOR_DIR not set; original shaders remain external");
        return;
    };
    let path = std::path::PathBuf::from(dir).join("compute-e30970f1727063fd.cso");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let shader = translate(&bytes).expect("dynamic structured-buffer shader should translate");
    assert!(shader.root_srvs.contains(&super::SLOT_STRIDES));
    let mut values = vec![0; shader.user_data_dwords() as usize];
    assert!(values.len() > 9);
    values[9] = 0x0800_1234;
    let srv: Vec<_> = shader.heap_demands.iter().filter(|d| d.class == 1)
        .map(|d| d.slot(&values)).collect();
    assert!(srv.contains(&Some(0x1234)), "original root-constant heap index: {:?}", shader.heap_demands);
    assert!(shader.heap_demands.iter().all(|d| d.slot(&values).is_some()),
        "original fixture should not need any whole-class scan: {:?}", shader.heap_demands);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for shader fixture");
    let mut dxil = shader.dxil.clone();
    validator.validate(&mut dxil).expect("dynamic structured-buffer DXIL validation");
}

/// These two original programs fetch their heap indices from buffer data,
/// not root constants. Their SRV demand must remain unknown.
#[test]
fn original_gpu_loaded_heap_indices_require_fallback() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    for hash in ["9209777e8face9dd", "42f717f1b4d18e94"] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir).join(format!("compute-{hash}.cso"))).unwrap();
        let shader = translate(&bytes).unwrap_or_else(|e| panic!("{hash}: {e}"));
        assert!(shader.root_srvs.contains(&super::SLOT_STRIDES));
        assert!(shader.heap_demands.iter().any(|d| d.class == 1 && d.code.is_empty()),
            "{hash}: a GPU-loaded heap index must retain fallback");
    }
}

/// Mesh 7972997d15fd9e69 reads the group identifier v5 through an original
/// v_readfirstlane instruction. The ID is uniform across the mesh group, so
/// its read can use the lane's value without a physical wave operation.
/// This does NOT establish correctness of the other wave64 operations, nor
/// authorize the host to disable its wave-size compatibility guard.
#[test]
fn original_mesh_group_uniform_readfirstlane_validates() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else {
        eprintln!("VICTORY_MESH_DESCRIPTOR_DIR not set; original mesh fixture stays external");
        return;
    };
    let path = std::path::PathBuf::from(dir).join("mesh-7972997d15fd9e69.cso");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let shader = translate(&bytes).expect("native original mesh group program should translate");
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for original mesh fixture");
    let mut dxil = shader.dxil.clone();
    validator.validate(&mut dxil).expect("original mesh DXIL validation");
}

/// Scene mesh c894/59b1: both original shader stages validate and can form
/// individual host pipelines, but the actual graphics pipeline is rejected
/// on wave32 hardware. Record the precise remaining native first-lane sites;
/// *never* clear the wave requirement merely to recover its draw calls.
#[test]
fn original_prison_scene_c894_mesh_wave64_blocker_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("mesh-c89489afd89ee7f0.cso");
    let bytes = std::fs::read(&path).expect("external scene mesh fixture");
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
    assert!(native.wave64);
    // Match the production translator: bounded M0 register ranges may change
    // the SSA graph and the last-lane values of the indexed reads.
    let (p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let parts = super::lower::Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
        outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
        prims: c.part(crate::dxbc::FourCC::PSG1)
            .map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
        mesh_stride: 128,
    };
    let all: Vec<_> = p.blocks.iter().flat_map(|bb| &bb.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32")
        .map(|i| i.machine.offset).collect();
    let waterfall: Vec<_> = p.blocks.iter().flat_map(|bb| &bb.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32"
            && i.writes.first().is_some_and(|w| res.waterfall.reads.contains(&w.1)))
        .map(|i| i.machine.offset).collect();
    eprintln!("c894 all firstlanes {all:x?}; waterfall {waterfall:x?}");
    eprintln!("c894 arithmetic {:?}", super::lower::uniform_arithmetic_firstlane_audit(&parts));
    eprintln!("c894 scalar select {:x?}", super::lower::uniform_scalar_select_firstlane_audit(&parts, true));
    eprintln!("c894 uniform fetch {:?}", super::lower::mesh_fetch_firstlane_audit(&parts));
    eprintln!("c894 broadcasts {:x?}", super::lower::mesh_broadcast_audit(&parts));
    assert_eq!(super::lower::relative_read_waterfalls_audit(&parts), [0x1858, 0x1880, 0x18a8]);
    // Every independently certified loop must fail closed when its original
    // index, M0 feed, retry, or destination-register phi is altered.
    // Mutate *one* native SSA/CFG fact at a time so unrelated loops remain
    // certifiable; this is a direct negative test of the actual fixture.
    let audited = |p: &crate::ir::Program| super::lower::relative_read_waterfalls_fixture_audit(p, &st);
    for &offset in &[0x1858usize, 0x1880, 0x18a8] {
        let b = p.blocks.iter().position(|bb| bb.offset == offset).unwrap();
        let other: Vec<_> = [0x1858, 0x1880, 0x18a8].into_iter().filter(|&x| x != offset).collect();
        let reject = |modified: crate::ir::Program| {
            let found = audited(&modified);
            assert!(!found.contains(&offset), "invalid mutated waterfall {offset:#x} accepted: {found:x?}");
            assert_eq!(found, other, "independent waterfalls changed unexpectedly");
        };
        let mut modified = p.clone();
        let index_read = modified.blocks[b].insts[1].reads.iter_mut()
            .find(|(r, _)| *r == crate::ir::Reg::V(64)).unwrap();
        index_read.1 = crate::ir::Value(u32::MAX - 1);
        reject(modified);
        let mut modified = p.clone();
        let m0_read = modified.blocks[b].insts[3].reads.iter_mut()
            .find(|(r, _)| *r == crate::ir::Reg::S(2)).unwrap();
        m0_read.1 = crate::ir::Value(u32::MAX - 2);
        reject(modified);
        let mut modified = p.clone();
        modified.blocks[b].exit = crate::ir::Exit::Jump(b + 1);
        reject(modified);
        let mut modified = p.clone();
        let back = modified.blocks[b].preds.iter().position(|&pred| pred == b).unwrap();
        let phi = modified.blocks[b].phis.iter_mut()
            .find(|phi| matches!(phi.reg, crate::ir::Reg::V(_))).unwrap();
        phi.args[back] = crate::ir::Value(u32::MAX - 3);
        reject(modified);
        let mut modified = p.clone();
        let target = modified.blocks[b].insts[4].writes.iter_mut()
            .find(|(reg, _)| matches!(reg, crate::ir::Reg::V(_))).unwrap();
        target.1 = crate::ir::Value(u32::MAX - 4);
        reject(modified);
    }
    // Three *different* waterfall loops perform indexed VGPR reads. Audit
    // their actual SSA and CFG rather than assuming descriptor-waterfall
    // certification covers these relative-register operations.
    let loops = [
        (0x1858usize, 0x1868usize, 0x1870usize, 0x1874usize, 27u8, 57u8),
        (0x1880, 0x1890, 0x1898, 0x189c, 28, 60),
        (0x18a8, 0x18b8, 0x18c0, 0x18c4, 29, 61),
    ];
    for &(first, relative, retry, restore, src, dst) in &loops {
        let (b, bb) = p.blocks.iter().enumerate()
            .find(|(_, bb)| bb.insts.first().is_some_and(|i| i.machine.offset == first))
            .expect("relative waterfall loop header");
        let names: Vec<_> = bb.insts.iter().map(|i| (i.machine.offset, i.name())).collect();
        eprintln!("c894 relative loop {first:#x} block={b} preds={:?} phis={:?} exit={:?} insts={names:x?}",
            bb.preds, bb.phis.iter().map(|ph| ph.reg).collect::<Vec<_>>(), bb.exit);
        assert!(matches!(bb.exit, crate::ir::Exit::Branch { taken, not_taken }
            if taken == b && p.blocks[not_taken].insts.first().is_some_and(|i| i.machine.offset == restore)));
        assert_eq!(names.first().unwrap().0, first);
        assert_eq!(names.last().unwrap().0, retry);
        let at = |offset| bb.insts.iter().find(|i| i.machine.offset == offset)
            .expect("relative waterfall instruction");
        let firstlane = at(first);
        let index = firstlane.read(crate::ir::Reg::V(64)).unwrap();
        let selected = firstlane.write(crate::ir::Reg::S(2)).unwrap();
        assert_eq!(at(first+4).read(crate::ir::Reg::V(64)), Some(index));
        assert_eq!(at(first+4).read(crate::ir::Reg::S(2)), Some(selected));
        assert_eq!(at(first+12).read(crate::ir::Reg::S(2)), Some(selected));
        let mov = at(relative);
        assert_eq!(mov.name(), "v_movrels_b32");
        assert!(mov.read(crate::ir::Reg::V(src)).is_some());
        assert!(mov.write(crate::ir::Reg::V(dst)).is_some());
        eprintln!("c894 relative {relative:#x} m0={:?} index={index:?} selected={selected:?} guards={:?}",
            mov.read(crate::ir::Reg::S(crate::ir::M0)), st.guards[b]);
    }
    let lowered = super::lower::lower(&parts).unwrap();
    eprintln!("c894 emitted wave_ops={} op_names={:?}", lowered.wave_ops, lowered.op_names.iter().filter(|n| n.contains("wave")).collect::<Vec<_>>());
    assert!(!lowered.wave_ops, "certified per-lane relative reads must eliminate the wave64-only operations");
    assert!(lowered.op_names.iter().all(|name| !name.contains("wave")));
    let built = crate::dxbc::Container::parse(&lowered.dxil).unwrap();
    let flags = built.part(crate::dxbc::FourCC::SFI0)
        .expect("production SFI0 feature metadata");
    let features = u64::from_le_bytes(flags.data.try_into().unwrap());
    assert_eq!(features & 0x4000, 0, "the host must see no wave intrinsic feature");
    let psv = built.part(crate::dxbc::FourCC::PSV0)
        .expect("production PSV0 metadata");
    assert!(psv.data.len() >= 24, "mesh shader PSV0 metadata");
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    let mut dxil = lowered.dxil;
    validator.validate(&mut dxil).unwrap();
}

#[test]
fn original_prison_scene_spilled_compute_root_table_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    for hash in ["3b90165a77d95cdf", "caa78b8ca62bc3d6", "28918b7150a7aad5", "f6e47efa50226beb"] {
        let path = std::path::PathBuf::from(&dir).join(format!("compute-{hash}.cso"));
        let bytes = std::fs::read(path).expect("original prison compute shader fixture");
        let (spilled, loads, rewritten) = crate::spill::original_fixture_spill_audit(&bytes);
        eprintln!("{hash} spill={spilled} scalar_loads={loads} rewritten={rewritten}");
        assert!(spilled, "the fixture needs spilled root data");
        assert!(rewritten > 0, "the original table loads must be recognized");
    }
}

/// The original mesh's scalar join at offset 0xf0 receives two independently
/// uniform operands via scalar-controlled, wave-uniform branches. This and
/// the four other proven broadcasts should emit no host wave intrinsics.
/// DXIL validation is NOT a GPU result comparison; enabling execution still
/// requires separate mesh-output verification. Native fixture stays external.
#[test]
fn original_mesh_797_proven_uniform_broadcasts_need_no_wave_ops() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("mesh-7972997d15fd9e69.cso");
    let bytes = std::fs::read(path).expect("external native mesh fixture");
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let mut defs = std::collections::HashSet::new();
    defs.extend(p.inputs.iter().map(|&(_, v)| v));
    defs.extend(p.blocks.iter().flat_map(|b| b.phis.iter().map(|p| p.value)));
    defs.extend(p.blocks.iter().flat_map(|b| b.insts.iter().flat_map(|i| i.writes.iter().map(|&(_, v)| v))));
    let missing: Vec<_> = p.blocks.iter().flat_map(|b| b.insts.iter())
        .flat_map(|i| i.reads.iter().filter(|(_, v)| !defs.contains(v)).map(move |&(r, v)| (i.machine.offset, r, v)))
        .collect();
    assert!(missing.is_empty(), "native SSA must have complete definitions: {missing:?}");
    let st = crate::structure::structure(&p).unwrap();
    for b in &p.blocks {
        if let Some(phi) = b.phis.iter().find(|phi| phi.value == crate::ir::Value(72)) {
            assert_eq!((phi.reg, b.preds.len(), phi.args.len()), (crate::ir::Reg::S(4), 2, 2));
            for &pred in &b.preds {
                assert!(st.bools.support(st.reach[pred]).iter().all(|a|
                    matches!(a, crate::structure::Atom::Uniform(_))));
            }
        }
    }
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let l = super::lower::lower(&super::lower::Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
        outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
        prims: c.part(crate::dxbc::FourCC::PSG1)
            .map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
        mesh_stride: 128,
    }).unwrap();
    assert!(!l.wave_ops, "proven native broadcasts must not emit host wave operations");
    assert!(l.op_names.iter().all(|name| !name.contains("wave")));
    assert!(l.insts.iter().all(|inst| !inst.contains("k118,")));
}

/// Inspect the native scalar descriptor consumed by the original 910b mesh
/// program's scalar buffer loads. Fixture stays outside public source.
#[test]
fn original_mesh_910b_scalar_buffer_descriptor_use_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("mesh-910b4be059bfb28f.cso");
    let bytes = std::fs::read(path).expect("external original mesh fixture");
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let writer = p.blocks.iter().flat_map(|b| &b.insts)
        .find(|i| i.machine.offset == 0xfe0).expect("descriptor load at 0xfe0");
    let values: Vec<_> = writer.writes.iter().map(|&(_,v)| v).collect();
    let mut uses = Vec::new();
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        if inst.reads.iter().any(|&(_,v)| values.contains(&v)) {
            uses.push((inst.machine.offset, inst.name()));
        }
    }
    eprintln!("original mesh 910b descriptor uses: {uses:?}");
    assert_eq!(uses, [
        (0xfec, "s_buffer_load_dwordx4"),
        (0xff4, "s_buffer_load_dwordx4"),
        (0x1000, "s_buffer_load_dword"),
    ]);
    // The fatal EXEC-dependent branch at 0x1098 ultimately depends on
    // a *different* descriptor: t97, whose third dword is read as the
    // record count at 0xed8 and multiplied by its 80-byte stride.
    let record = p.blocks.iter().flat_map(|b| &b.insts)
        .find(|i| i.machine.offset == 0xec0).expect("record-count descriptor");
    let count = record.write(crate::ir::Reg::S(6)).expect("descriptor dword 2");
    let read = p.blocks.iter().flat_map(|b| &b.insts)
        .find(|i| i.machine.offset == 0xed8).expect("record-count consumer");
    assert_eq!(read.read(crate::ir::Reg::S(6)), Some(count));
    assert!(native.used.iter().any(|u| u.kind == crate::native::ResourceType::Srv
        && u.root == crate::native::RootType::Table && u.dwords == 4
        && u.lower_bound == 97 && u.table_offset_dwords == 0
        && u.stride == 80 && u.user_data == 9 && u.slot == 1));
    let shader = translate(&bytes).expect("native mesh 910b scalar SRV descriptor should translate");
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for original mesh fixture");
    let mut dxil = shader.dxil.clone();
    validator.validate(&mut dxil).expect("native mesh 910b DXIL validation");
}

#[test]
fn original_mesh_910b_uniform_arithmetic_firstlane_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("mesh-910b4be059bfb28f.cso");
    let bytes = std::fs::read(path).expect("external original mesh fixture");
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let parts = super::lower::Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
        outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
        prims: c.part(crate::dxbc::FourCC::PSG1)
            .map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
        mesh_stride: 128,
    };
    let audit = super::lower::uniform_arithmetic_firstlane_audit(&parts);
    let recovered: Vec<_> = audit.iter().filter(|(_, safe)| *safe).map(|(offset, _)| *offset).collect();
    assert_eq!(recovered, [0x12c, 0x184, 0x218], "{audit:x?}");
    for offset in [0x380, 0x384] {
        assert!(audit.contains(&(offset, false)), "divergent count must retain broadcast");
    }
    let broadcasts = super::lower::mesh_broadcast_audit(&parts);
    assert_eq!(broadcasts, [(0x380, 0), (0x384, 2)], "{broadcasts:x?}");
    let without = super::lower::uniform_scalar_select_firstlane_audit(&parts, false);
    let selects = super::lower::uniform_scalar_select_firstlane_audit(&parts, true);
    assert_eq!(without, [], "unproven scalar broadcast cannot justify late selects");
    assert_eq!(selects, [0x43cc, 0x43f4, 0x43f8], "{selects:x?}");
    let mut waterfalls: Vec<_> = p.blocks.iter().flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_readfirstlane_b32"
            && i.writes.first().is_some_and(|w| res.waterfall.reads.contains(&w.1)))
        .map(|i| i.machine.offset).collect();
    waterfalls.sort_unstable();
    let mut expected = vec![0xdf8];
    for start in [0x2800, 0x2890, 0x28f0, 0x3b88, 0x3bf0, 0x4020, 0x4088, 0x40f0] {
        expected.extend((0..4).map(|i| start + 4 * i));
    }
    assert_eq!(waterfalls, expected, "{waterfalls:x?}");
    let lowered = super::lower::lower(&parts).unwrap();
    assert!(lowered.op_names.iter().any(|name| name.contains("barrier")), "shared broadcast requires a group barrier");
    assert!(lowered.nonuniform_handles > 0, "waterfalls must retain nonuniform handles");
    assert!(!lowered.wave_ops, "original shader still uses host wave ops: {:?}", lowered.op_names);
    assert!(!lowered.op_names.iter().any(|name| name.contains("wave")));
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for original mesh fixture");
    let mut dxil = lowered.dxil;
    validator.validate(&mut dxil).expect("native mesh 910b DXIL validation");
}

#[test]
fn original_mesh_partial_reciprocal_firstlane_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    use crate::gfx10::{Fields, Operand};
    use crate::ir::{Reg, EXEC};
    use crate::structure::Bool;
    for (hash, offsets, full, counts, selects, extra) in [
        ("3a087a87f2a2ad3c", [0x2b8, 0x2c0, 0x2c4, 0x2cc, 0x2d0],
            [0xec, 0x144], [0x3f8, 0x3fc], [0xab4, 0xadc, 0xae0], false),
        ("8f690b0f5dbbce02", [0x2f8, 0x300, 0x304, 0x30c, 0x310],
            [0x138, 0x190], [0x438, 0x43c], [0xdc4, 0xdec, 0xdf0], true),
    ] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir).join(format!("mesh-{hash}.cso"))).unwrap();
        let c = crate::dxbc::Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let parts = super::lower::Parts {
            native: &native, p: &p, st: &st, io: &io, res: &res,
            inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
            outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
            prims: c.part(crate::dxbc::FourCC::PSG1).map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
            mesh_stride: 128,
        };
        let site = |offset| p.blocks.iter().enumerate().find_map(|(b, bb)|
            bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
        let (b, read_i) = site(offsets[4]);
        let read = &p.blocks[b].insts[read_i];
        let exec = (read.read(Reg::S(EXEC)).unwrap(), read.read(Reg::S(EXEC + 1)).unwrap());
        let mask = st.mask_bits(exec.0, Some(exec.1)).expect("known partial EXEC");
        assert_ne!(mask, Bool::TRUE);
        assert_ne!(mask, Bool::FALSE);
        for offset in offsets {
            let (bb, i) = site(offset);
            assert_eq!(bb, b);
            let inst = &p.blocks[bb].insts[i];
            assert_eq!((inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC+1))), (Some(exec.0), Some(exec.1)));
            assert!(st.guards[bb][i].is_none());
        }
        assert_eq!(super::lower::partial_reciprocal_firstlane_audit(&parts), [offsets[4]], "{hash}");
        let original_full: Vec<_> = super::lower::uniform_arithmetic_firstlane_audit(&parts)
            .into_iter().filter(|&(_, yes)| yes).map(|(offset, _)| offset).collect();
        assert_eq!(original_full, full, "{hash}");
        assert_eq!(super::lower::mesh_broadcast_audit(&parts), [(counts[0], 0), (counts[1], 2)], "{hash}");
        assert_eq!(super::lower::uniform_scalar_select_firstlane_audit(&parts, true), selects, "{hash}");
        let reject = |label: &str, bad_p: &crate::ir::Program, bad_st: &crate::structure::Structured| {
            let neg = super::lower::Parts { native: &native, p: bad_p, st: bad_st, io: &io, res: &res,
                inputs: parts.inputs.clone(), outputs: parts.outputs.clone(), prims: parts.prims.clone(), mesh_stride: 128 };
            assert!(!super::lower::partial_reciprocal_firstlane_audit(&neg).contains(&offsets[4]), "{hash}: {label}");
        };
        let (bb, first) = site(offsets[0]);
        let mut guarded = st.clone();
        guarded.guards[bb][first] = Some(mask);
        reject("additional writer guard", &p, &guarded);
        let mut mismatched = p.clone();
        let (eb, ei) = site(0x3c);
        mismatched.blocks[b].insts[first].reads.iter_mut().find(|(r, _)| *r == Reg::S(EXEC)).unwrap().1 =
            p.blocks[eb].insts[ei].write(Reg::S(EXEC)).unwrap();
        reject("different EXEC", &mismatched, &st);
        let mut scale = p.clone();
        let (_, mul) = site(offsets[2]);
        let Fields::Valu(v) = &mut scale.blocks[b].insts[mul].fields else { panic!() };
        *v.src.iter_mut().find(|op| **op == Operand::Literal(0x4f7ffffe)).unwrap() = Operand::Literal(0x3f800000);
        reject("different reciprocal scale", &scale, &st);
        let mut divergent = p.clone();
        let Fields::Valu(v) = &p.blocks[b].insts[first].fields else { panic!() };
        let Operand::Scalar { code, .. } = v.src[0] else { panic!() };
        divergent.blocks[b].insts[first].reads.iter_mut().find(|(r, _)| *r == Reg::S(code)).unwrap().1 =
            p.inputs.iter().find(|(r, _)| *r == Reg::V(5)).unwrap().1;
        reject("lane-derived divisor", &divergent, &st);
        let mut escape = p.clone();
        let (reg, result) = read.writes[0];
        let other = p.blocks.iter().enumerate().find(|(bb, block)| *bb > b && !block.insts.is_empty()).unwrap().0;
        escape.blocks[other].insts[0].reads.push((reg, result));
        reject("scalar escape", &escape, &st);
        let mut phi_escape = p.clone();
        let phi = phi_escape.blocks.iter_mut().skip(b + 1).flat_map(|bb| &mut bb.phis)
            .find(|phi| !phi.args.is_empty()).expect("later join phi");
        phi.args[0] = result;
        reject("phi escape", &phi_escape, &st);
        let mut corrected_escape = p.clone();
        let correction = p.blocks[b].insts.iter().skip(read_i + 1).find(|inst| inst.name() == "s_add_i32").unwrap();
        let &(r, v) = correction.writes.iter().find(|(r, _)| matches!(r, Reg::S(0..=105))).unwrap();
        corrected_escape.blocks[other].insts[0].reads.push((r, v));
        reject("transitive scalar escape", &corrected_escape, &st);
        let mut local_mask = st.clone();
        let terminal = p.blocks[b].insts.iter().enumerate().skip(read_i + 1).find(|(_, inst)| inst.name() == "v_mul_hi_u32").unwrap().0;
        local_mask.guards[b][terminal] = Some(mask);
        reject("guarded terminal use", &p, &local_mask);
        let data_audit = super::lower::partial_data_conversion_firstlane_audit(&parts);
        if extra {
            assert_eq!(data_audit, [0xc24, 0xc70], "{hash}");
            use crate::resource::{Role, Source};
            use crate::native::ResourceType;
            let (source_b, source_i) = site(0xad8);
            let source = &p.blocks[source_b].insts[source_i];
            let uniform_float = source.write(Reg::S(6)).unwrap();
            let (desc_b, desc_i) = site(0x81c);
            let desc = &p.blocks[desc_b].insts[desc_i];
            let (input_desc_b, input_desc_i) = site(0x840);
            let source_descriptor = res.descriptors.iter().find(|d| d.site == (source_b, source_i) && d.role == Role::Resource).unwrap();
            assert_eq!(source_descriptor.source, Source::Loaded { load: (input_desc_b, input_desc_i), first: 0 });
            assert!(native.used.iter().any(|u| u.kind == ResourceType::Srv && u.user_data == 8
                && u.table_offset_dwords == 384 && u.lower_bound == 71 && u.stride == 96 && u.dwords == 4));
            for (cv_off, read_off, mul_off, add_off, load_off, move_off, vr) in [
                (0xc18, 0xc24, 0xc28, 0xc2c, 0xc30, 0xc3c, 28),
                (0xc64, 0xc70, 0xc74, 0xc7c, 0xc80, 0xc8c, 29),
            ] {
                let (b, ri) = site(read_off);
                let (cv_b, ci) = site(cv_off);
                let (mb, mi) = site(mul_off);
                let (ab, ai) = site(add_off);
                let (lb, li) = site(load_off);
                let (vb, vi) = site(move_off);
                assert_eq!([cv_b, mb, ab, lb, vb], [b; 5]);
                let firstlane = &p.blocks[b].insts[ri];
                let convert = &p.blocks[b].insts[ci];
                let load = &p.blocks[b].insts[li];
                let exec = (firstlane.read(Reg::S(EXEC)).unwrap(), firstlane.read(Reg::S(EXEC+1)).unwrap());
                let mask = st.mask_bits(exec.0, Some(exec.1)).unwrap();
                assert_ne!(mask, Bool::TRUE);
                assert_ne!(mask, Bool::FALSE);
                for idx in [ci, ri, vi, vi + 1, vi + 2] {
                    let w = &p.blocks[b].insts[idx];
                    assert_eq!((w.read(Reg::S(EXEC)), w.read(Reg::S(EXEC+1))), (Some(exec.0), Some(exec.1)));
                    assert!(st.guards[b][idx].is_none());
                }
                assert_eq!(convert.read(Reg::S(6)), Some(uniform_float));
                assert_eq!(convert.write(Reg::V(vr)), firstlane.read(Reg::V(vr)));
                let result = firstlane.write(Reg::S(0)).unwrap();
                assert_eq!(p.blocks[b].insts[mi].read(Reg::S(0)), Some(result));
                let addr = p.blocks[b].insts[ai].write(Reg::S(14)).unwrap();
                assert_eq!(load.read(Reg::S(14)), Some(addr));
                for r in 48..52 {
                    assert_eq!(load.read(Reg::S(r)), desc.write(Reg::S(r)));
                }
                let descriptor = res.descriptors.iter().find(|d| d.site == (b, li) && d.role == Role::Resource).unwrap();
                assert_eq!(descriptor.source, Source::Loaded { load: (desc_b, desc_i), first: 0 });
                let reject = |label: &str, bad_p: &crate::ir::Program, bad_st: &crate::structure::Structured,
                    bad_res: &crate::resource::Resources, bad_native: &crate::native::NativeProgram| {
                    let neg = super::lower::Parts { native: bad_native, p: bad_p, st: bad_st, io: &io, res: bad_res,
                        inputs: parts.inputs.clone(), outputs: parts.outputs.clone(), prims: parts.prims.clone(), mesh_stride: 128 };
                    assert!(!super::lower::partial_data_conversion_firstlane_audit(&neg).contains(&read_off),
                        "{hash} {read_off:x}: {label}");
                };
                let mut varying = p.clone();
                varying.blocks[b].insts[ci].reads.iter_mut().find(|(r, _)| *r == Reg::S(6)).unwrap().1 =
                    p.inputs.iter().find(|(r, _)| *r == Reg::V(5)).unwrap().1;
                reject("lane-dependent conversion source", &varying, &st, &res, &native);
                let mut guarded = st.clone();
                guarded.guards[b][ci] = Some(mask);
                reject("extra conversion guard", &p, &guarded, &res, &native);
                let mut wrong_exec = p.clone();
                let (eb, ei) = site(0x3c);
                wrong_exec.blocks[b].insts[vi].reads.iter_mut().find(|(r, _)| *r == Reg::S(EXEC)).unwrap().1 =
                    p.blocks[eb].insts[ei].write(Reg::S(EXEC)).unwrap();
                reject("different terminal EXEC", &wrong_exec, &st, &res, &native);
                let mut wrong_scale = p.clone();
                let Fields::Sop { src, .. } = &mut wrong_scale.blocks[b].insts[mi].fields else { panic!() };
                src[1] = Operand::Const(141);
                reject("wrong coefficient stride", &wrong_scale, &st, &res, &native);
                if read_off == 0xc70 {
                    let (kb, ki) = site(0xc78);
                    assert_eq!(kb, b);
                    let k = &p.blocks[b].insts[ki];
                    assert_eq!(k.name(), "s_add_i32");
                    assert!(matches!(&k.fields, Fields::Sop { src, .. }
                        if src.as_slice() == [Operand::Scalar { code: 0, dwords: 1 }, Operand::Const(176)]));
                    assert_eq!(k.read(Reg::S(0)), p.blocks[b].insts[mi].write(Reg::S(0)));
                    assert_eq!(k.write(Reg::S(0)), p.blocks[b].insts[ai].read(Reg::S(0)));
                    let mut wrong_bias = p.clone();
                    let Fields::Sop { src, .. } = &mut wrong_bias.blocks[b].insts[ki].fields else { panic!() };
                    src[1] = Operand::Const(177);
                    reject("wrong +48 coefficient bias", &wrong_bias, &st, &res, &native);
                    let mut scc_escape = p.clone();
                    scc_escape.blocks[b].insts[ai].reads.push((Reg::Scc, k.write(Reg::Scc).unwrap()));
                    reject("bias SCC escape", &scc_escape, &st, &res, &native);
                }
                let mut wrong_offset = p.clone();
                let Fields::Smem { offset, .. } = &mut wrong_offset.blocks[b].insts[li].fields else { panic!() };
                *offset = 4;
                reject("different data load offset", &wrong_offset, &st, &res, &native);
                let mut varying_descriptor = p.clone();
                varying_descriptor.blocks[b].insts[li].reads.iter_mut().find(|(r, _)| *r == Reg::S(48)).unwrap().1 =
                    p.inputs.iter().find(|(r, _)| *r == Reg::V(5)).unwrap().1;
                reject("unproven descriptor SSA", &varying_descriptor, &st, &res, &native);
                let mut dynamic_descriptor = res.clone();
                dynamic_descriptor.scalar.iter_mut().find(|(s, _)| *s == (desc_b, desc_i)).unwrap().1.dynamic = Some(result);
                reject("dynamic descriptor table index", &p, &st, &dynamic_descriptor, &native);
                let mut writable = native.clone();
                writable.used.iter_mut().find(|u| u.user_data == 8 && u.table_offset_dwords == 384).unwrap().kind = ResourceType::Uav;
                reject("writable resource", &p, &st, &res, &writable);
                let other = p.blocks.iter().enumerate().find(|(bb, block)| *bb > b && !block.insts.is_empty()).unwrap().0;
                for (label, r, value) in [("conversion escape", Reg::S(0), result),
                    ("address escape", Reg::S(14), addr), ("loaded data escape", Reg::S(2), load.write(Reg::S(2)).unwrap())] {
                    let mut escaped = p.clone();
                    escaped.blocks[other].insts[0].reads.push((r, value));
                    reject(label, &escaped, &st, &res, &native);
                }
                let mut phi_escape = p.clone();
                let phi = phi_escape.blocks.iter_mut().skip(b + 1).flat_map(|bb| &mut bb.phis)
                    .find(|phi| !phi.args.is_empty()).expect("later join phi");
                phi.args[0] = load.write(Reg::S(0)).unwrap();
                reject("loaded scalar phi escape", &phi_escape, &st, &res, &native);
            }
        } else {
            assert!(data_audit.is_empty(), "{hash}: {data_audit:x?}");
        }
        let lowered = super::lower::lower(&parts).unwrap();
        assert!(!lowered.wave_ops, "{hash}: {:?}", lowered.op_names);
        assert!(!lowered.op_names.iter().any(|n| n.contains("wave")), "{hash}");
        let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
            .expect("DXIL validator required for original mesh fixture");
        let mut dxil = lowered.dxil;
        validator.validate(&mut dxil).unwrap();
        let mut dxil = translate(&bytes).unwrap().dxil;
        let host = crate::dxbc::Container::parse(&dxil).unwrap();
        let feature_bytes: [u8; 8] = host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap();
        assert_eq!(u64::from_le_bytes(feature_bytes) & 0x4000, 0, "{hash}");
        validator.validate(&mut dxil).unwrap();
    }
}

#[test]
fn original_mesh_coefficient_table_variants() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    use crate::gfx10::{Fields, Operand};
    use crate::ir::{Reg, EXEC};
    use crate::native::ResourceType;
    use crate::resource::{Role, Source};
    use crate::structure::Bool;
    for hash in ["9bfc2a16c35cb95b", "82585ef9bcdc4f9f", "db4835a364449a45", "b7a81f5b96f4ca93", "37f363f0a5dcd806", "ba8d52f4ea1a1bac", "9214ed391b7f8483", "c962201cddbdd7dd", "e135eaf39576263a", "74a70fe506cab1ae"] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir).join(format!("mesh-{hash}.cso"))).unwrap();
        let c = crate::dxbc::Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let parts = super::lower::Parts {
            native: &native, p: &p, st: &st, io: &io, res: &res,
            inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
            outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
            prims: c.part(crate::dxbc::FourCC::PSG1).map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
            mesh_stride: 128,
        };
        let data = super::lower::partial_data_conversion_firstlane_audit(&parts);
        let recip = super::lower::partial_reciprocal_firstlane_audit(&parts);
        let full = super::lower::uniform_arithmetic_firstlane_audit(&parts);
        let counts = super::lower::mesh_broadcast_audit(&parts);
        let selects = super::lower::uniform_scalar_select_firstlane_audit(&parts, true);
        let fetch = super::lower::mesh_fetch_firstlane_audit(&parts);
        let full_sites: Vec<_> = full.into_iter().filter(|&(_, ok)| ok).map(|(o, _)| o).collect();
        let fetch_sites: Vec<_> = fetch.into_iter().filter(|&(_, ok)| ok).map(|(o, _)| o).collect();
        let site = |offset| p.blocks.iter().enumerate().find_map(|(b, bb)|
            bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
        let inst = |offset| {
            let (b, i) = site(offset);
            &p.blocks[b].insts[i]
        };
        let wave_free = true;
        if wave_free {
            let (source_off, cbv_off, srv_off, scalar_reg, descriptor_reg, load_offset, read_sites,
                full_expected, recip_expected, count_expected, select_expected, cv_regs, address_reg, first_reg):
                (_, _, _, _, _, _, _, &[usize], &[usize], &[(usize, u32)], _, _, _, _) = match hash {
                "9bfc2a16c35cb95b" => (0x978, 0x8e8, 0x8a8, 52, 36, 44, [0xc94, 0xce0],
                    &[0x138, 0x190], &[0x310], &[(0x438, 0), (0x43c, 2)], [0xe38, 0xe60, 0xe64], [8, 8], 1, 0),
                "82585ef9bcdc4f9f" => (0xb30, 0x850, 0x82c, 6, 36, 36, [0xc24, 0xc70],
                    &[0x138, 0x190], &[0x310], &[(0x438, 0), (0x43c, 2)], [0xdc8, 0xdf0, 0xdf4], [23, 36], 1, 0),
                "db4835a364449a45" => (0x1070, 0x1038, 0x1040, 12, 0, 44, [0x10b0, 0x10fc],
                    &[0x138, 0x190, 0x21c], &[], &[(0x378, 0), (0x37c, 2)], [0x1234, 0x125c, 0x1260], [8, 8], 5, 4),
                "b7a81f5b96f4ca93" => (0x5f0, 0x5a4, 0x5bc, 46, 40, 0, [0x8d4, 0x920],
                    &[], &[], &[], [0xa28, 0xa50, 0xa54], [3, 27], 1, 0),
                "37f363f0a5dcd806" => (0x65c, 0x634, 0x66c, 46, 40, 0, [0x8a0, 0x8ec],
                    &[], &[], &[], [0x9d0, 0x9f8, 0x9fc], [5, 25], 1, 0),
                "ba8d52f4ea1a1bac" => (0x624, 0x5f4, 0x8e0, 22, 0, 0, [0x93c, 0x98c],
                    &[], &[], &[], [0xab8, 0xae0, 0xae4], [18, 19], 5, 4),
                "9214ed391b7f8483" => (0x7f0, 0x764, 0xc54, 68, 16, 40, [0xdac, 0xdf8],
                    &[0x138, 0x190, 0x224], &[], &[(0x380, 0), (0x384, 2)], [0xf60, 0xf88, 0xf8c], [20, 35], 14, 0),
                "c962201cddbdd7dd" => (0x600, 0x5dc, 0x610, 46, 40, 0, [0x844, 0x890],
                    &[], &[], &[], [0x974, 0x99c, 0x9a0], [7, 23], 1, 0),
                "e135eaf39576263a" => (0x628, 0x5f4, 0x934, 42, 0, 0, [0x9d4, 0xa24],
                    &[], &[], &[], [0xb74, 0xb9c, 0xba0], [20, 21], 5, 4),
                "74a70fe506cab1ae" => (0x5e8, 0x5a4, 0x5bc, 46, 40, 0, [0x8d4, 0x920],
                    &[], &[], &[], [0xa28, 0xa50, 0xa54], [7, 31], 1, 0),
                _ => unreachable!(),
            };
            assert_eq!(data, read_sites, "{hash}");
            assert_eq!(full_sites, full_expected, "{hash}");
            assert_eq!(recip, recip_expected, "{hash}");
            assert_eq!(counts, count_expected, "{hash}");
            assert_eq!(selects, select_expected, "{hash}");
            match hash {
                "b7a81f5b96f4ca93" | "37f363f0a5dcd806" | "c962201cddbdd7dd" | "74a70fe506cab1ae" => assert_eq!(fetch_sites, [0x13c]),
                "ba8d52f4ea1a1bac" | "e135eaf39576263a" => assert_eq!(fetch_sites, [0x154]),
                _ => assert!(fetch_sites.is_empty(), "{hash}"),
            }
            let source_site = site(source_off);
            let source = inst(source_off);
            if load_offset == 0 {
                assert_eq!(source.name(), "s_buffer_load_dwordx4");
                assert!(matches!(&source.fields, Fields::Smem {
                    data: Some(Operand::Scalar { code, dwords: 4 }), offset: 0, .. } if *code + 2 == scalar_reg));
            } else {
                assert_eq!(source.name(), "s_buffer_load_dword");
            }
            assert!(matches!(&source.fields, Fields::Smem { offset, .. } if *offset == load_offset));
            let scalar = source.write(Reg::S(scalar_reg)).unwrap();
            let descriptor = |s| &res.descriptors.iter().find(|d| d.site == s && d.role == Role::Resource).unwrap().source;
            assert_eq!(*descriptor(source_site), Source::Loaded { load: site(cbv_off), first: 0 });
            assert!(native.used.iter().any(|u| u.kind == ResourceType::Cbv && u.user_data == 10
                && u.table_offset_dwords == 0 && u.lower_bound == 4 && u.dwords == 4));
            let srv_table = if matches!(hash, "ba8d52f4ea1a1bac" | "e135eaf39576263a") { 368 } else { 384 };
            let srv_register = if load_offset == 0 { 70 } else { 72 };
            assert!(native.used.iter().any(|u| u.kind == ResourceType::Srv && u.user_data == 8
                && u.table_offset_dwords == srv_table && u.lower_bound == srv_register && u.stride == 96 && u.dwords == 4));
            for (arm, read_off) in read_sites.into_iter().enumerate() {
                let (b, ri) = site(read_off);
                let (cv_b, ci) = site(read_off - 12);
                let (mb, mi) = site(read_off + 4);
                let (ab, ai) = site(read_off + if arm == 0 { 8 } else { 12 });
                let wait_before_load = if matches!(hash, "ba8d52f4ea1a1bac" | "e135eaf39576263a") { 4 } else { 0 };
                let (lb, li) = site(read_off + if arm == 0 { 12 } else { 16 } + wait_before_load);
                let (vb, vi) = site(read_off + if arm == 0 { 24 } else { 28 } + wait_before_load);
                assert_eq!([cv_b, mb, ab, lb, vb], [b; 5], "{hash}");
                let bb = &p.blocks[b];
                let read = &bb.insts[ri];
                let cv = &bb.insts[ci];
                let load = &bb.insts[li];
                let exec = (read.read(Reg::S(EXEC)).unwrap(), read.read(Reg::S(EXEC + 1)).unwrap());
                let mask = st.mask_bits(exec.0, Some(exec.1)).unwrap();
                assert_ne!(mask, Bool::TRUE);
                assert_ne!(mask, Bool::FALSE);
                for idx in [ci, ri, vi, vi + 1, vi + 2] {
                    let w = &bb.insts[idx];
                    assert_eq!((w.read(Reg::S(EXEC)), w.read(Reg::S(EXEC + 1))), (Some(exec.0), Some(exec.1)));
                    assert!(st.guards[b][idx].is_none());
                }
                assert_eq!(cv.read(Reg::S(scalar_reg)), Some(scalar));
                assert_eq!(cv.write(Reg::V(cv_regs[arm])), read.read(Reg::V(cv_regs[arm])));
                let first = read.write(Reg::S(first_reg)).unwrap();
                assert_eq!(bb.insts[mi].read(Reg::S(first_reg)), Some(first));
                let mul = bb.insts[mi].write(Reg::S(first_reg)).unwrap();
                if arm == 1 {
                    let bias = inst(read_off + 8);
                    assert_eq!(bias.name(), "s_add_i32");
                    assert!(matches!(&bias.fields, Fields::Sop { src, .. } if
                        src.as_slice() == [Operand::Scalar { code: first_reg, dwords: 1 }, Operand::Const(176)]));
                    assert_eq!(bias.read(Reg::S(first_reg)), Some(mul));
                    assert_eq!(bias.write(Reg::S(first_reg)), bb.insts[ai].read(Reg::S(first_reg)));
                } else {
                    assert_eq!(bb.insts[ai].read(Reg::S(first_reg)), Some(mul));
                }
                let addr = bb.insts[ai].write(Reg::S(address_reg)).unwrap();
                assert_eq!(load.read(Reg::S(address_reg)), Some(addr));
                assert_eq!(*descriptor((b, li)), Source::Loaded { load: site(srv_off), first: 0 });
                for r in descriptor_reg..descriptor_reg + 4 {
                    assert_eq!(load.read(Reg::S(r)), inst(srv_off).write(Reg::S(r)));
                }
                let reject = |label: &str, bad_p: &crate::ir::Program, bad_st: &crate::structure::Structured,
                    bad_res: &crate::resource::Resources, bad_native: &crate::native::NativeProgram| {
                    let neg = super::lower::Parts { native: bad_native, p: bad_p, st: bad_st, io: &io, res: bad_res,
                        inputs: parts.inputs.clone(), outputs: parts.outputs.clone(), prims: parts.prims.clone(), mesh_stride: 128 };
                    assert!(!super::lower::partial_data_conversion_firstlane_audit(&neg).contains(&read_off),
                        "{hash} {read_off:x}: {label}");
                };
                let mut wrong_cbv_offset = p.clone();
                let Fields::Smem { offset, .. } = &mut wrong_cbv_offset.blocks[source_site.0].insts[source_site.1].fields else { panic!() };
                *offset = 32;
                reject("unsupported CBV offset", &wrong_cbv_offset, &st, &res, &native);
                if load_offset == 0 {
                    // The x4 exception is specifically component 2, even
                    // though the other three components are scalar too.
                    for component in [0, 1, 3] {
                        let mut wrong_component = p.clone();
                        wrong_component.blocks[b].insts[ci].reads.iter_mut()
                            .find(|(r, _)| *r == Reg::S(scalar_reg)).unwrap().1 =
                            source.write(Reg::S(scalar_reg - 2 + component)).unwrap();
                        reject("unsupported x4 component", &wrong_component, &st, &res, &native);
                    }
                }
                let mut varying = p.clone();
                varying.blocks[b].insts[ci].reads.iter_mut().find(|(r, _)| *r == Reg::S(scalar_reg)).unwrap().1 =
                    p.inputs.iter().find(|(r, _)| *r == Reg::V(5)).unwrap().1;
                reject("lane-dependent input", &varying, &st, &res, &native);
                let mut guarded = st.clone();
                guarded.guards[b][ci] = Some(mask);
                reject("extra conversion guard", &p, &guarded, &res, &native);
                let mut wrong_exec = p.clone();
                wrong_exec.blocks[b].insts[vi].reads.iter_mut().find(|(r, _)| *r == Reg::S(EXEC)).unwrap().1 =
                    inst(0x3c).write(Reg::S(EXEC)).unwrap();
                reject("different terminal EXEC", &wrong_exec, &st, &res, &native);
                let mut dynamic = res.clone();
                dynamic.scalar.iter_mut().find(|(s, _)| *s == site(cbv_off)).unwrap().1.dynamic = Some(first);
                reject("dynamic CBV descriptor", &p, &st, &dynamic, &native);
                let mut writable = native.clone();
                writable.used.iter_mut().find(|u| u.user_data == 8 && u.table_offset_dwords == srv_table).unwrap().kind = ResourceType::Uav;
                reject("writable SRV", &p, &st, &res, &writable);
                let other = p.blocks.iter().enumerate().find(|(bb, block)| *bb > b && !block.insts.is_empty()).unwrap().0;
                let (reg, loaded) = load.writes[0];
                for (label, reg, value) in [("conversion escape", Reg::S(first_reg), first),
                    ("address escape", Reg::S(address_reg), addr), ("loaded data escape", reg, loaded)] {
                    let mut escaped = p.clone();
                    escaped.blocks[other].insts[0].reads.push((reg, value));
                    reject(label, &escaped, &st, &res, &native);
                }
                let mut phi_escape = p.clone();
                let phi = phi_escape.blocks.iter_mut().skip(b + 1).flat_map(|bb| &mut bb.phis)
                    .find(|phi| !phi.args.is_empty()).expect("later join phi");
                phi.args[0] = loaded;
                reject("loaded scalar phi escape", &phi_escape, &st, &res, &native);
            }
        }
        let lowered = super::lower::lower(&parts).unwrap();
        assert_eq!(!lowered.wave_ops, wave_free, "{hash}: {:?}", lowered.op_names);
        assert_eq!(!lowered.op_names.iter().any(|n| n.contains("wave")), wave_free, "{hash}");
        let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
        let mut dxil = lowered.dxil;
        validator.validate(&mut dxil).unwrap();
        let mut dxil = translate(&bytes).unwrap().dxil;
        let host = crate::dxbc::Container::parse(&dxil).unwrap();
        let feature_bytes: [u8; 8] = host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap();
        assert_eq!(u64::from_le_bytes(feature_bytes) & 0x4000 == 0, wave_free, "{hash}");
        validator.validate(&mut dxil).unwrap();
    }
}

#[test]
fn original_compute_f8_lds_allocation_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir).join("compute-f8f50ab85e7b10bb.cso")).unwrap();
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Cs).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    use crate::structure::LaneExit;
    use crate::gfx10::{Fields,Operand};
    use crate::ir::{Exit,Reg};
    let parts = super::lower::Parts { native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
    assert_eq!(super::lower::uniform_arithmetic_firstlane_audit(&parts), vec![(0x2978,true)]);
    assert_eq!(super::lower::lds_allocation_audit(&parts), Some(((0,10),222,224)));
    let site = |offset| p.blocks.iter().enumerate().find_map(|(b,bb)|
        bb.insts.iter().position(|i| i.machine.offset == offset).map(|i|(b,i))).unwrap();
    let reject = |why, p: &crate::ir::Program, st: &crate::structure::Structured, res: &crate::resource::Resources| {
        let mutated = super::lower::Parts { native: &native, p, st, io: &io, res,
            inputs:vec![],outputs:vec![],prims:vec![],mesh_stride:128 };
        assert_eq!(super::lower::lds_allocation_audit(&mutated),None,"{why}");
    };
    let (b,i) = site(0x24);
    let mut wrong_lane = p.clone();
    let Fields::Valu(v) = &mut wrong_lane.blocks[b].insts[i].fields else { panic!() };
    v.src[0] = Operand::Const(129);
    reject("nonzero initializing lane",&wrong_lane,&st,&res);
    let (b,i) = site(0x4c);
    let mut unguarded = st.clone();
    unguarded.guards[b][i] = None;
    reject("unconditional LDS clear",&p,&unguarded,&res);
    for offset in [0x4c,0x2e0,0x3e3c,0x3e54,0x3e88] {
        let (b,i) = site(offset);
        let mut wrong = p.clone();
        let Fields::Ds { offset0, .. } = &mut wrong.blocks[b].insts[i].fields else { panic!() };
        *offset0 ^= 4;
        reject("wrong allocator cell",&wrong,&st,&res);
    }
    let (b,i) = site(0x2d8);
    let mut wrong_step = p.clone();
    let Fields::Valu(v) = &mut wrong_step.blocks[b].insts[i].fields else { panic!() };
    v.src[0] = Operand::Const(130);
    reject("increment by two",&wrong_step,&st,&res);
    let mut early = p.clone();
    let mut early_st = st.clone();
    early.blocks[1].exit = Exit::End;
    early_st.exits[1] = LaneExit::End;
    reject("native exit before rendezvous",&early,&early_st,&res);
    let mut bypass = p.clone();
    let mut bypass_st = st.clone();
    bypass.blocks[223].exit = Exit::Jump(227);
    bypass_st.exits[223] = LaneExit::Jump(227);
    reject("leader skips publication",&bypass,&bypass_st,&res);
    let (b,i) = site(0x3e48);
    let mut no_return = p.clone();
    let Fields::Buffer { cache, .. } = &mut no_return.blocks[b].insts[i].fields else { panic!() };
    cache.glc = false;
    reject("atomic does not return old base",&no_return,&st,&res);
    let (b,i) = site(0x3eac);
    let mut overlap = p.clone();
    let Fields::Buffer { offset, .. } = &mut overlap.blocks[b].insts[i].fields else { panic!() };
    *offset = 36;
    reject("overlapping record stores",&overlap,&st,&res);
    let (b,i) = site(0x3e74);
    let mut dynamic = res.clone();
    dynamic.scalar.iter_mut().find(|(s,_)| *s == (b,i)).unwrap().1.dynamic = Some(p.inputs[0].1);
    reject("dynamic output view",&p,&st,&dynamic);
    let (b,i) = site(0x2978);
    let mut varying = p.clone();
    varying.blocks[b].insts[i].reads.iter_mut().find(|(r,_)| *r == Reg::V(41)).unwrap().1 =
        io.inputs.iter().find(|(_,v)| *v == crate::io::Input::ThreadIdInGroup(0)).unwrap().0;
    reject("varying firstlane source",&varying,&st,&res);
    let lowered = super::lower::lower(&parts).unwrap();
    assert!(!lowered.wave_ops);
    let barriers = lowered.insts.iter().filter(|i| i.contains("k80,")).collect::<Vec<_>>();
    assert_eq!(barriers.len(),3);
    assert_eq!(barriers.iter().filter(|i| i.contains("[k80, k9]")).count(),2);
    assert_eq!(barriers.iter().filter(|i| i.contains("[k80, k11]")).count(),1);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    let mut dxil = lowered.dxil;
    validator.validate(&mut dxil).unwrap();
    let mut dxil = translate(&bytes).unwrap().dxil;
    validator.validate(&mut dxil).unwrap();
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    let flags = u64::from_le_bytes(host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap());
    assert_eq!(flags & 0x4000, 0);
    let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()),0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()),u32::MAX);
}

#[test]
fn original_compute_8284_hierarchical_reduction_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir).join("compute-82849f7a614208eb.cso")).unwrap();
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Cs).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let audit = |p: &crate::ir::Program, st: &crate::structure::Structured| {
        let parts = super::lower::Parts { native: &native,p,st,io:&io,res:&res,
            inputs:vec![],outputs:vec![],prims:vec![],mesh_stride:128 };
        super::lower::logical_lane_audit(&parts)
    };
    let plan = audit(&p,&st);
    let expected = [0x94,0xc0,0x110,0x128,0x16c,0x17c];
    if plan.is_empty() {
        eprintln!("8284 IO {:?}",native.io);
        for (b,bb) in p.blocks.iter().enumerate() {
            eprintln!("block {b} @{:x} reach {:?} exit {:?}", bb.offset,st.reach[b],st.exits[b]);
            for (i,w) in bb.insts.iter().enumerate().filter(|(_,w)| w.name().starts_with("ds_") || w.name().starts_with("v_mbcnt") || w.name().starts_with("v_readlane")) {
                eprintln!(" {:x} {} {:?} guard {:?} reads {:?}",w.machine.offset,w.name(),w.fields,st.guards[b][i],w.reads);
            }
        }
    }
    assert_eq!(plan.iter().map(|x|x.0).collect::<Vec<_>>(),expected);
    assert!(plan.iter().all(|&(_,_,publish)|publish));
    assert_eq!(plan.iter().map(|x|x.1).collect::<std::collections::HashSet<_>>().len(),expected.len());
    use crate::structure::{Bool,LaneExit};
    use crate::gfx10::{Fields,Operand};
    let (b,i) = p.blocks.iter().enumerate().find_map(|(b,bb)|bb.insts.iter().position(|w|w.machine.offset==0x110).map(|i|(b,i))).unwrap();
    let mut bypass = st.clone(); bypass.exits[0] = LaneExit::End;
    assert!(audit(&p,&bypass).is_empty());
    let mut partial = st.clone(); partial.guards[b][i] = Some(Bool::FALSE);
    assert!(audit(&p,&partial).is_empty());
    let mut looped = st.clone(); looped.exits[b] = LaneExit::Jump(b);
    assert!(audit(&p,&looped).is_empty());
    let mut dynamic = p.clone();
    let v = dynamic.blocks.iter_mut().flat_map(|bb|&mut bb.insts).find(|w|w.name()=="v_mbcnt_lo_u32_b32").unwrap();
    let Fields::Valu(v) = &mut v.fields else { panic!() };
    v.src[0] = Operand::Scalar {code:0,dwords:1};
    assert!(audit(&dynamic,&st).is_empty());
    let parts = super::lower::Parts { native:&native,p:&p,st:&st,io:&io,res:&res,
        inputs:vec![],outputs:vec![],prims:vec![],mesh_stride:128 };
    let lower = super::lower::lower(&parts).unwrap();
    assert!(!lower.wave_ops,"{:?}",lower.op_names);
    assert_eq!(lower.insts.iter().filter(|i|i.contains("[k80, k11]")).count(),6);
    let mut dxil = translate(&bytes).unwrap().dxil;
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    assert_eq!(u64::from_le_bytes(host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap()) & 0x4000,0);
    let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()),0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()),u32::MAX);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut dxil).unwrap();
}

#[test]
fn original_compute_13df_logical_lane_snapshot_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir).join("compute-13dfdd3b1136e407.cso")).unwrap();
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Cs).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    use crate::gfx10::{Fields, Operand};
    use crate::ir::{Exit, Reg};
    use crate::structure::{Bool, LaneExit};
    let parts = super::lower::Parts { native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
    let crate::native::StageIo::Cs(cs) = native.io else { panic!() };
    assert!(native.wave64);
    assert_eq!(cs.threads, [8,8,1]);
    assert_eq!(cs.lds_bytes(), 0);
    let plan = super::lower::logical_lane_audit(&parts);
    let expected = [0xbc,0xc4,0xcc,0xd4,0xdc,0xe4,0xec,0xf4,0xfc,0x104,0x10c,0x114,
        0x174,0x184,0x18c,0x194,0x19c,0x1a4,0x1b8,0x1d8,0x1e0,0x1fc,0x204,0x20c,
        0x254,0x25c,0x27c,0x28c,0x294,0x29c,0x2a4,0x2ac,0x2b4,0x2bc,0x2cc,0x2dc,
        0x2f0,0x2f8,0x304,0x310];
    assert_eq!(plan.iter().map(|x| x.0).collect::<Vec<_>>(), expected);
    let mut published = std::collections::HashSet::new();
    for &(_, base, publish) in &plan {
        assert_eq!(base % 64, 0);
        if publish { assert!(published.insert(base)); } else { assert!(published.contains(&base)); }
    }
    assert!(published.len() < plan.len());
    let reject = |why, p: &crate::ir::Program, st: &crate::structure::Structured, native: &crate::native::NativeProgram| {
        let mutated = super::lower::Parts { native, p, st, io: &io, res: &res,
            inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
        assert!(super::lower::logical_lane_audit(&mutated).is_empty(), "{why}");
    };
    let (b,i) = p.blocks.iter().enumerate().find_map(|(b,bb)|
        bb.insts.iter().position(|i| i.machine.offset == 0x254).map(|i|(b,i))).unwrap();
    let mut dynamic = p.clone();
    let Fields::Valu(v) = &mut dynamic.blocks[b].insts[i].fields else { panic!() };
    v.src[1] = Operand::Scalar { code: 6, dwords: 1 };
    reject("dynamic lane",&dynamic,&st,&native);
    let mut wrong_lane = p.clone();
    let Fields::Valu(v) = &mut wrong_lane.blocks[b].insts[i].fields else { panic!() };
    v.src[1] = Operand::Literal(64);
    reject("out of range lane",&wrong_lane,&st,&native);
    let mut masked = st.clone();
    masked.guards[b][i] = Some(Bool::FALSE);
    reject("masked operation",&p,&masked,&native);
    let mut exec = p.clone();
    exec.blocks[b].insts[i].writes.push((Reg::S(126),p.inputs[0].1));
    reject("EXEC mutation",&exec,&st,&native);
    let mut looped = p.clone(); let mut loop_st = st.clone();
    looped.blocks[b].exit = Exit::Jump(b); loop_st.exits[b] = LaneExit::Jump(b);
    reject("revisited snapshot",&looped,&loop_st,&native);
    let mut multi = native.clone();
    let crate::native::StageIo::Cs(cs) = &mut multi.io else { panic!() };
    cs.threads = [8,16,1];
    reject("multiple waves",&p,&st,&multi);
    let mut lds = native.clone();
    let crate::native::StageIo::Cs(cs) = &mut lds.io else { panic!() };
    cs.rsrc2 |= 1 << 15;
    reject("native LDS",&p,&st,&lds);
    let lowered = super::lower::lower(&parts).unwrap();
    assert!(!lowered.wave_ops);
    assert!(!lowered.op_names.iter().any(|n| n.contains("wave")));
    assert_eq!(lowered.insts.iter().filter(|i| i.contains("[k80, k9]")).count(),published.len());
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    let mut dxil = lowered.dxil;
    validator.validate(&mut dxil).unwrap();
    let mut dxil = translate(&bytes).unwrap().dxil;
    validator.validate(&mut dxil).unwrap();
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    let flags = u64::from_le_bytes(host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap());
    assert_eq!(flags & 0x4000, 0);
    let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()),0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()),u32::MAX);
}

#[test]
fn original_mesh_f525_uniform_fetch_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("mesh-f5257487f2945fdb.cso");
    let bytes = std::fs::read(path).expect("external original mesh fixture");
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Ms).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let parts = super::lower::Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: super::signature(c.part(crate::dxbc::FourCC::ISG1).unwrap().data).unwrap(),
        outputs: super::signature(c.part(crate::dxbc::FourCC::OSG1).unwrap().data).unwrap(),
        prims: c.part(crate::dxbc::FourCC::PSG1)
            .map(|x| super::signature(x.data).unwrap()).unwrap_or_default(),
        mesh_stride: 128,
    };
    use crate::ir::{Reg, EXEC};
    use crate::structure::{Atom, Bool, LaneExit};
    let audit = super::lower::mesh_fetch_firstlane_audit(&parts);
    assert_eq!(audit, [(0x54, false), (0xfc, true), (0x644, true),
        (0xa08, false), (0xa30, false), (0xa34, false)]);
    assert_eq!(super::lower::uniform_scalar_select_firstlane_audit(&parts, true), [0xa08, 0xa30, 0xa34]);
    assert!(super::lower::mesh_broadcast_audit(&parts).is_empty());
    let site = |offset| p.blocks.iter().enumerate().find_map(|(b, bb)|
        bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
    let (b, mov) = site(0x634);
    let (fb, fetch) = site(0x638);
    let (rb, read) = site(0x644);
    assert_eq!((fb, rb), (b, b));
    let inst = |i: usize| &p.blocks[b].insts[i];
    let exec = |i| (inst(i).read(Reg::S(EXEC)).unwrap(), inst(i).read(Reg::S(EXEC + 1)).unwrap());
    assert_eq!(exec(mov), exec(fetch));
    assert_eq!(exec(fetch), exec(read));
    assert_ne!(st.mask_bits(exec(read).0, Some(exec(read).1)), Some(Bool::TRUE));
    assert!([mov, fetch, read].into_iter().all(|i| st.guards[b][i].is_none()));
    assert_eq!(inst(mov).write(Reg::V(13)), inst(fetch).read(Reg::V(13)));
    assert_eq!(inst(fetch).write(Reg::V(14)), inst(read).read(Reg::V(14)));
    let (head, _) = site(0x59c);
    let (arm, _) = site(0x5a0);
    let (join, _) = site(0x5b8);
    assert_eq!(p.blocks[join].preds, [head, arm]);
    assert_eq!(p.blocks[arm].preds, [head]);
    assert_eq!(st.exits[arm], LaneExit::Jump(join));
    let LaneExit::Branch { when, taken, not_taken } = st.exits[head] else { panic!("expected diamond") };
    assert_eq!((taken, not_taken), (join, arm));
    let (cmp_b, cmp_i) = site(0x4f0);
    let scc = p.blocks[cmp_b].insts[cmp_i].write(Reg::Scc).unwrap();
    assert_eq!(st.bools.support(when), [Atom::Uniform(scc)]);
    let rejected = |bad_p: &crate::ir::Program, bad_st: &crate::structure::Structured| {
        let negative = super::lower::Parts { native: &native, p: bad_p, st: bad_st,
            io: &io, res: &res, inputs: parts.inputs.clone(), outputs: parts.outputs.clone(),
            prims: parts.prims.clone(), mesh_stride: 128 };
        assert!(super::lower::mesh_fetch_firstlane_audit(&negative).contains(&(0x644, false)));
    };
    // A divergent choice at the exact same phi cannot justify a scalar index.
    let mut divergent = st.clone();
    let (lane_branch, _) = site(0x5f4);
    let LaneExit::Branch { when: lane_when, .. } = st.exits[lane_branch] else { panic!() };
    assert!(st.bools.support(lane_when).iter().any(|a| matches!(a, Atom::Lane(_))));
    divergent.exits[head] = LaneExit::Branch { when: lane_when, taken, not_taken };
    rejected(&p, &divergent);
    // Same SSA but an additional guard cannot certify every active lane.
    let mut guarded = st.clone();
    guarded.guards[b][fetch] = Some(lane_when);
    rejected(&p, &guarded);
    // A different EXEC pair, a lane-derived descriptor, or an escaping SGPR
    // must each leave the wave read in place.
    let mut mismatch = p.clone();
    mismatch.blocks[b].insts[read].reads.iter_mut().find(|(r, _)| *r == Reg::S(EXEC)).unwrap().1 =
        p.blocks[lane_branch].insts.iter().find(|i| i.machine.offset == 0x5ec).unwrap().read(Reg::S(EXEC)).unwrap();
    rejected(&mismatch, &st);
    let mut varying = p.clone();
    varying.blocks[b].insts[fetch].reads.iter_mut().find(|(r, _)| *r == Reg::S(4)).unwrap().1 = inst(mov).write(Reg::V(13)).unwrap();
    rejected(&varying, &st);
    let mut escaping = p.clone();
    let (outside_b, outside_i) = site(0x970);
    escaping.blocks[outside_b].insts[outside_i].reads.push((Reg::S(16), inst(read).write(Reg::S(16)).unwrap()));
    rejected(&escaping, &st);
    let lowered = super::lower::lower(&parts).unwrap();
    assert!(!lowered.wave_ops, "original shader still uses host wave ops: {:?}", lowered.op_names);
    assert!(lowered.op_names.iter().all(|name| !name.contains("wave")));
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("DXIL validator required for original mesh fixture");
    let mut dxil = lowered.dxil;
    validator.validate(&mut dxil).expect("original mesh f525 DXIL validation");
    let mut dxil = translate(&bytes).expect("original mesh f525 translation").dxil;
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    let feature_bytes: [u8; 8] = host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap();
    assert_eq!(u64::from_le_bytes(feature_bytes) & 0x4000, 0, "final container must not require host wave ops");
    validator.validate(&mut dxil).expect("original mesh f525 final container validation");
}

#[test]
fn original_compute_static_table_cbv_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir).join("compute-13dfdd3b1136e407.cso")).unwrap();
    let original = translate(&bytes).unwrap();
    let mut optimized = super::translate_static_cbvs(&bytes).unwrap();
    assert_eq!(optimized.static_table_cbvs, [super::StaticTableCbv {slot:0,table_offset_dwords:40}]);
    assert!(optimized.root_srvs.contains(&super::STATIC_CBV_SCRATCH));
    assert!(original.static_table_cbvs.is_empty());
    assert_ne!(optimized.dxil,original.dxil);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut optimized.dxil).unwrap();
}

#[test]
fn original_compute_separable_filter_lds_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    for (hash, rendezvous) in [("3d002bca51069e01",0x2f8), ("f753697c1c03772b",0x2fc)] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir).join(format!("compute-{hash}.cso"))).unwrap();
        let c = crate::dxbc::Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Cs).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let parts = super::lower::Parts { native: &native, p: &p, st: &st, io: &io, res: &res,
            inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
        let proof = super::lower::filter_lds_audit(&parts);
        if proof.is_none() {
            eprintln!("{hash}: native IO {:?}; inputs {:?}", native.io, io.inputs);
            for (b, bb) in p.blocks.iter().enumerate() {
                eprintln!("block {b} @{:x} preds {:?} exit {:?} reach {:?}", bb.offset, bb.preds, st.exits[b], st.reach[b]);
                for (i, w) in bb.insts.iter().enumerate().filter(|(_,w)| w.name().starts_with("ds_") || w.name().contains("cmpx") || w.name().starts_with("s_cmp_eq") || w.name() == "image_store") {
                    eprintln!("  {:x} {} {:?} guard {:?} reads {:?} writes {:?}", w.machine.offset, w.name(), w.fields, st.guards[b][i], w.reads, w.writes);
                }
            }
        }
        assert_eq!(proof, Some((rendezvous,320,320)), "{hash}");
        let lower = super::lower::lower(&parts).unwrap();
        assert!(!lower.wave_ops, "{hash}: {:?}", lower.op_names);
        let barriers = lower.insts.iter().filter(|i| i.contains("k80,")).collect::<Vec<_>>();
        assert_eq!(barriers.len(), 1);
        assert!(barriers[0].contains("[k80, k11]"), "{barriers:?}");
        let mut dxil = translate(&bytes).unwrap().dxil;
        let host = crate::dxbc::Container::parse(&dxil).unwrap();
        let flags: [u8; 8] = host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap();
        assert_eq!(u64::from_le_bytes(flags) & 0x4000, 0);
        let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
        assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()), u32::MAX);
        let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
        validator.validate(&mut dxil).unwrap();

        // No unsafe relaxation for an aliasing producer, changed loop bound,
        // uninitialized halo read, skipped rendezvous or non-unique output.
        use crate::gfx10::{Fields, Operand};
        use crate::ir::Reg;
        use crate::structure::{Bool, LaneExit};
        let reject = |label: &str, pp: &crate::ir::Program, ss: &crate::structure::Structured| {
            let negative = super::lower::Parts { native: &native, p: pp, st: ss, io: &io, res: &res,
                inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
            assert_eq!(super::lower::filter_lds_audit(&negative), None, "{hash}: {label}");
        };
        let site = |name: &str, ordinal: usize| p.blocks.iter().enumerate().flat_map(|(b, bb)|
            bb.insts.iter().enumerate().filter(move |(_, w)| w.name() == name).map(move |(i, _)| (b, i))).nth(ordinal).unwrap();
        let (b,i) = site("ds_write2_b32",1);
        let mut alias = p.clone();
        let Fields::Ds { offset0, offset1, .. } = &mut alias.blocks[b].insts[i].fields else { panic!() };
        *offset0 = 0; *offset1 = 1;
        reject("overlapping producer", &alias, &st);
        let mut guard = st.clone(); guard.guards[b][i] = Some(Bool::FALSE);
        reject("guarded producer", &p, &guard);
        let (b,i) = site("ds_read2_b32",0);
        let mut read = p.clone();
        let Fields::Ds { offset0, .. } = &mut read.blocks[b].insts[i].fields else { panic!() };
        *offset0 = 255;
        reject("out of range read", &read, &st);
        let loop_b = p.blocks.iter().enumerate().find(|(b, bb)| bb.exit.successors().contains(b)).unwrap().0;
        let cmp_i = p.blocks[loop_b].insts.iter().position(|w| w.name() == "s_cmp_eq_u32").unwrap();
        let mut bound = p.clone();
        let Fields::Sop { src, .. } = &mut bound.blocks[loop_b].insts[cmp_i].fields else { panic!() };
        src[1] = Operand::Const(138);
        reject("three loop iterations", &bound, &st);
        let (b,i) = site("image_store",0);
        let mut duplicate = p.clone();
        duplicate.blocks[b].insts[i].reads.iter_mut().find(|(r, _)| *r == Reg::V(2)).unwrap().1 = p.inputs.iter().find(|(r, _)| *r == Reg::V(0)).unwrap().1;
        reject("unproven image coordinate", &duplicate, &st);
        let mut bypass = st.clone(); bypass.exits[0] = LaneExit::End;
        reject("rendezvous bypass", &p, &bypass);
    }
}

#[test]
fn original_compute_a25ed_implicit_lds_audit() {
    let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir).join("compute-a25ed0e3d4d521ba.cso")).unwrap();
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(crate::dxbc::FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == crate::native::HwStage::Cs).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let parts = super::lower::Parts { native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
    assert_eq!(super::lower::implicit_lds_audit(&parts), Some(0x154));
    assert_eq!(super::lower::implicit_lds_coverage(&parts), Some((160, 160)));
    use crate::structure::{Bool, LaneExit};
    use crate::ir::{Reg, EXEC};
    use crate::gfx10::{Fields, Operand};
    let site = |offset| p.blocks.iter().enumerate().find_map(|(b, bb)|
        bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
    assert_eq!(site(0x150), (2, 0));
    let restore = &p.blocks[2].insts[0];
    assert_eq!(st.mask_bits(restore.write(Reg::S(EXEC)).unwrap(), restore.write(Reg::S(EXEC+1))), Some(Bool::TRUE));
    assert_eq!(st.reach[2], Bool::TRUE);
    let reject = |label: &str, pp: &crate::ir::Program, ss: &crate::structure::Structured| {
        let negative = super::lower::Parts { native: &native, p: pp, st: ss, io: &io, res: &res,
            inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
        assert_eq!(super::lower::implicit_lds_audit(&negative), None, "{label}");
    };
    let LaneExit::Branch { when, .. } = st.exits[0] else { panic!("expected halo branch") };
    let mut bypass = st.clone();
    bypass.exits[0] = LaneExit::Branch { when, taken: 4, not_taken: 1 };
    reject("barrier bypass", &p, &bypass);
    let mut cycle = st.clone();
    cycle.exits[1] = LaneExit::Jump(1);
    reject("loop", &p, &cycle);
    let mut partial = st.clone();
    partial.reach[2] = when;
    reject("partial convergence", &p, &partial);
    let (wb, wi) = site(0x100);
    let (wb2, wi2) = site(0x10c);
    let (rb, ri) = site(0x1d0);
    let mut pre_read = p.clone();
    pre_read.blocks[wb].insts[wi] = p.blocks[rb].insts[ri].clone();
    reject("read before barrier", &pre_read, &st);
    let mut post_write = p.clone();
    post_write.blocks[rb].insts[ri] = p.blocks[wb].insts[wi].clone();
    reject("write after barrier", &post_write, &st);
    let mut guarded = st.clone();
    guarded.guards[wb][wi] = Some(when);
    reject("unknown extra writer guard", &p, &guarded);
    let mut alias = p.clone();
    let inst = &mut alias.blocks[wb2].insts[wi2];
    let Fields::Ds { addr, offset0, offset1, .. } = &mut inst.fields else { panic!() };
    *addr = Some(Operand::Vector { reg: 12, dwords: 1 });
    *offset0 = 4;
    *offset1 = 0;
    *inst.reads.iter_mut().find(|(r, _)| *r == Reg::V(10)).unwrap() =
        (Reg::V(12), p.blocks[wb].insts[wi].read(Reg::V(12)).unwrap());
    reject("overlapping writes", &alias, &st);
    let mut uncovered = p.clone();
    let Fields::Ds { offset0, .. } = &mut uncovered.blocks[rb].insts[ri].fields else { panic!() };
    *offset0 = 0;
    reject("read without a producer", &uncovered, &st);
    let mut oob = p.clone();
    let Fields::Ds { offset1, .. } = &mut oob.blocks[rb].insts[ri].fields else { panic!() };
    *offset1 = 255;
    reject("out of bounds", &oob, &st);
    let mut unknown = p.clone();
    let (ab, ai) = site(0xf4);
    let Fields::Valu(v) = &mut unknown.blocks[ab].insts[ai].fields else { panic!() };
    v.src[0] = Operand::Scalar { code: 9, dwords: 1 };
    reject("unproven address operand", &unknown, &st);
    let mut crosslane = p.clone();
    let mut words = Vec::new();
    for w in [0xD7600008u32, 0x00014101, 0xBF810000] { words.extend(w.to_le_bytes()); }
    let inst = crate::gfx10::decode(&words).unwrap().remove(0);
    assert_eq!(inst.name(), Some("v_readlane_b32"));
    crosslane.blocks[2].insts[1].fields = inst.fields(true);
    crosslane.blocks[2].insts[1].machine = inst;
    reject("cross-lane read", &crosslane, &st);
    let mut image_alias = p.clone();
    let (sb, si) = site(0x290);
    image_alias.blocks[sb].insts[si].reads.iter_mut().find(|(r, _)| *r == Reg::V(3)).unwrap().1 =
        p.inputs.iter().find(|(r, _)| *r == Reg::V(0)).unwrap().1;
    reject("image read/store coordinate mismatch", &image_alias, &st);
    let lowered = super::lower::lower(&parts).unwrap();
    assert!(!lowered.wave_ops, "{:?}", lowered.op_names);
    assert!(lowered.op_names.iter().any(|n| n.contains("barrier")));
    assert!(lowered.op_names.iter().all(|n| !n.contains("wave")));
    let barriers = lowered.insts.iter().filter(|i| i.contains("k80,")).collect::<Vec<_>>();
    assert_eq!(barriers.len(), 1);
    assert!(barriers[0].contains("[k80, k9]"), "{barriers:?}");
    let mut dxil = translate(&bytes).unwrap().dxil;
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    let feature_bytes: [u8; 8] = host.part(crate::dxbc::FourCC::SFI0).unwrap().data.try_into().unwrap();
    assert_eq!(u64::from_le_bytes(feature_bytes) & 0x4000, 0);
    let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()), u32::MAX);
    assert_eq!(&psv[40..52], &[4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]);
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut dxil).unwrap();
    // Rebuild real SSA/Structured from a valid native mutation, not just
    // diagnostic IR changes. The second store now aliases the first store
    // at every thread's center cell, so the original WaveSize(64) must stay.
    let mut alias_native = native.clone();
    assert_eq!(&alias_native.code[0x10c..0x114], &[0xac, 0x00, 0x34, 0xd8, 0x0a, 0x0b, 0x00, 0x00]);
    alias_native.code[0x10c..0x114].copy_from_slice(&[0x04, 0x00, 0x34, 0xd8, 0x0c, 0x0b, 0x00, 0x00]);
    let ap = crate::ir::build(&alias_native.patched_code(), alias_native.wave64, alias_native.vgprs).unwrap();
    let ast = crate::structure::structure(&ap).unwrap();
    let ar = crate::resource::track(&ap, &alias_native.layout, alias_native.stage, Some(&ast)).unwrap();
    let aio = crate::io::raise(&ap, &alias_native).unwrap();
    let alias_parts = super::lower::Parts { native: &alias_native, p: &ap, st: &ast, io: &aio, res: &ar,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 128 };
    assert_eq!(super::lower::implicit_lds_audit(&alias_parts), None);
    let rejected = super::lower::lower(&alias_parts).unwrap();
    assert!(!rejected.op_names.iter().any(|n| n.contains("barrier")));
    let mut dxil = rejected.dxil;
    let host = crate::dxbc::Container::parse(&dxil).unwrap();
    let psv = host.part(crate::dxbc::FourCC::PSV0).unwrap().data;
    assert_eq!(u32::from_le_bytes(psv[20..24].try_into().unwrap()), 64);
    assert_eq!(u32::from_le_bytes(psv[24..28].try_into().unwrap()), 64);
    validator.validate(&mut dxil).unwrap();
}
