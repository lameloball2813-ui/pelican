//! Regression for original compute programs whose scans and LDS phases
//! still require a physical 64-lane wave. Fixtures remain external.

use super::{translate, translate_static_cbvs, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, StageIo};

/// Audit the actual escaping definitions, not merely the presence of a
/// recognized shape on a potentially dead temporary. Include phi-edge uses.
#[test]
fn original_prison_scan_live_ssa_outputs() {
    use crate::ir::{Reg, EXEC};
    use super::wave::{sections, Op, Shape};
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    for (hash, delta, sum_reg, saved_reg, join) in [
        ("28918b7150a7aad5", 0, 27, 12, 0x7e8),
        ("caa78b8ca62bc3d6", 4, 41, 22, 0x9b0),
    ] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir)
            .join(format!("compute-{hash}.cso"))).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == HwStage::Cs).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let entry = &p.blocks[0];
        let ss = sections(&p, 0);
        for (begin, end, output, reg, prefix) in [
            (0x94, 0xe0, 0xd8, Reg::S(sum_reg), false),
            (0xec, 0x158, 0x154, Reg::V(2), true),
            (0x16c, 0x1d8, 0x1d4, Reg::V(2), true),
        ] {
            let start = entry.insts.iter().position(|i| i.machine.offset == begin + delta).unwrap();
            let sec = ss.get(&start).expect("scan must remain in entry block");
            assert_eq!(entry.insts[sec.end].machine.offset, end + delta);
            let save = &entry.insts[start];
            let restore = &entry.insts[sec.end];
            let saved = save.writes.iter().find(|(r, _)| matches!(r, Reg::S(n) if *n < 106)).unwrap();
            assert_eq!(restore.read(saved.0), Some(saved.1), "restored mask must be the saved SSA value");
            assert!(restore.write(Reg::S(EXEC)).is_some());
            let want = entry.insts.iter().find(|i| i.machine.offset == output + delta).unwrap().write(reg).unwrap();
            let mut seen = false;
            for def in &entry.insts[start+1..sec.end] {
                for &(r, v) in &def.writes {
                    let mut uses = Vec::new();
                    for (b, bb) in p.blocks.iter().enumerate() {
                        for phi in &bb.phis {
                            if phi.args.contains(&v) { uses.push((b, bb.offset, true)); }
                        }
                        for (i, user) in bb.insts.iter().enumerate() {
                            if b == 0 && i > start && i < sec.end { continue; }
                            if user.reads.iter().any(|&(_, a)| a == v) {
                                uses.push((b, user.machine.offset, false));
                            }
                        }
                    }
                    if uses.is_empty() { continue; }
                    eprintln!("{hash} scan {:#x} def {:#x} {r:?} {v:?} shape={:?} escapes={uses:?}", begin+delta, def.machine.offset, sec.shapes.get(&v));
                    if sec.shapes.get(&v).is_none() && matches!(r, Reg::V(_)) {
                        for &(ub, offset, phi) in &uses {
                            if phi {
                                eprintln!("{hash} unknown {v:?}: phi escape block {ub}; cannot discard without transitive proof");
                                continue;
                            }
                            let user = p.blocks[ub].insts.iter().find(|i| i.machine.offset == offset).unwrap();
                            let operands = crate::ir::operand_reads(&p, user);
                            let explicit = user.reads.iter().any(|&(reg, value)| value == v && operands.contains(&reg));
                            let next_section = if ub == 0 { ss.iter().find(|(s, next)| entry.insts[**s].machine.offset < offset && offset < entry.insts[next.end].machine.offset).map(|(s, _)| entry.insts[*s].machine.offset) } else { None };
                            eprintln!("{hash} unknown {v:?} -> {offset:#x} {} explicit={explicit} subsequent_section={next_section:?}", user.name());
                            if !explicit {
                                assert!(user.reads.iter().filter(|&&(_, value)| value == v).all(|&(reg, _)| matches!(reg, Reg::V(_)) && user.write(reg).is_some()), "implicit read must be destination preservation");
                            }
                        }
                    }
                    if v == want {
                        seen = true;
                        assert!(matches!(sec.shapes.get(&v), Some(Shape::Prefix(Op::AddU, _, m, 0, false)) if prefix && *m == saved.1)
                            || matches!(sec.shapes.get(&v), Some(Shape::Active(Op::AddU, _, m, 0)) if !prefix && *m == saved.1));
                    }
                }
            }
            assert!(seen, "{hash}: expected output {output:#x} must escape section");
            eprintln!("{hash} scan {:#x} saved={saved:?} input_exec={:?} guard={:?}", begin+delta, save.read(Reg::S(EXEC)), st.guards[0][start]);
        }
        let (b, bb) = p.blocks.iter().enumerate().find(|(_, bb)| bb.offset == join).unwrap();
        let restore = &bb.insts[0];
        assert_eq!(restore.name(), "s_mov_b64");
        let join_saved = if delta == 0 { saved_reg } else { 0 };
        assert!(restore.read(Reg::S(join_saved)).is_some());
        eprintln!("{hash} LDS join block={b} reach={:?} preds={:?} saved={:?} guard={:?}", st.reach[b], bb.preds, restore.read(Reg::S(join_saved)), st.guards[b][0]);
        for &pred in &bb.preds {
            eprintln!("{hash} predecessor {pred} offset={:#x} reach={:?} exit={:?}", p.blocks[pred].offset, st.reach[pred], st.exits[pred]);
        }
        // A path to the join is not a convergence proof: the existing
        // once-only rendezvous certificate rejects any pre-join cycle.
        let mut todo = vec![0usize];
        let mut visited = std::collections::HashSet::new();
        let mut back_edges = Vec::new();
        while let Some(n) = todo.pop() {
            if n == b || !visited.insert(n) { continue; }
            for t in p.blocks[n].exit.successors() {
                if t != b && p.blocks[t].offset <= p.blocks[n].offset { back_edges.push((p.blocks[n].offset, p.blocks[t].offset)); }
                todo.push(t);
            }
        }
        assert!(!back_edges.is_empty(), "original producer loops must remain explicit proof obligations");
        eprintln!("{hash} pre-join native backward edges={back_edges:x?}");
    }
}

#[test]
fn original_prison_scan_shapes_and_lds_join_audit() {
    use super::wave::{sections, Op, Shape};
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    for (hash, delta, join, read) in [
        ("28918b7150a7aad5", 0, 0x7e8, 0x804),
        ("caa78b8ca62bc3d6", 4, 0x9b0, 0x9c4),
    ] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir)
            .join(format!("compute-{hash}.cso"))).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == HwStage::Cs).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        for (offset, prefix) in [(0x94 + delta, false), (0xec + delta, true), (0x16c + delta, true)] {
            let (b, i) = p.blocks.iter().enumerate().find_map(|(b, bb)|
                bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
            let ss = sections(&p, b);
            let sec = ss.get(&i).expect("whole-wave section");
            let proven = sec.shapes.values().any(|s| match s {
                Shape::Active(Op::AddU, _, _, 0) => !prefix,
                Shape::Prefix(Op::AddU, _, _, 0, false) => prefix,
                _ => false,
            });
            assert!(proven, "{hash} {offset:#x}: expected symbolic scan shape");
            eprintln!("{hash} scan {offset:#x}: block={b} reach={:?} guard={:?}", st.reach[b], st.guards[b][i]);
        }
        for offset in [join, read] {
            let (b, i) = p.blocks.iter().enumerate().find_map(|(b, bb)|
                bb.insts.iter().position(|i| i.machine.offset == offset).map(|i| (b, i))).unwrap();
            eprintln!("{hash} LDS {offset:#x}: block={b} preds={:?} reach={:?} guard={:?}", p.blocks[b].preds, st.reach[b], st.guards[b][i]);
        }
    }
}

#[test]
fn original_prison_scan_lds_preserves_wave64_requirement() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    for hash in ["28918b7150a7aad5", "caa78b8ca62bc3d6"] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir)
            .join(format!("compute-{hash}.cso"))).expect("external original compute fixture");
        let original = Container::parse(&bytes).unwrap();
        let native = crate::native_programs(original.part(FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|p| p.stage == HwStage::Cs).unwrap();
        assert!(native.wave64, "{hash}: native wave size");
        let StageIo::Cs(cs) = native.io else { panic!("{hash}: not compute") };
        assert_eq!(cs.threads, [64, 1, 1], "{hash}: group size");
        assert!(cs.lds_bytes() > 0, "{hash}: native LDS is required");

        for emit in [translate, translate_static_cbvs] {
            let mut dxil = emit(&bytes).unwrap().dxil;
            let host = Container::parse(&dxil).unwrap();
            let psv = host.part(FourCC::PSV0).unwrap().data;
            assert!(u32::from_le_bytes(psv[0..4].try_into().unwrap()) >= 48);
            let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
            let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
            assert_eq!((min, max), (64, 64), "{hash}: do not admit a 32-lane host without a full scan/LDS proof");
            let flags = u64::from_le_bytes(host.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
            assert_ne!(flags & 0x4000, 0, "{hash}: native wave operations remain");
            assert_eq!(&psv[40..52], &[64, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
            let mut v = validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
            v.validate(&mut dxil).unwrap();
        }
    }
}
