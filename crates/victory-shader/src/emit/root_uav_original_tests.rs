//! Numerical regression for the original spilled-root UAV fill shader.
use super::host::{Dispatch, Host, Root};
use crate::{Container, FourCC, HwStage, RootType, native::StageIo};

#[test]
fn original_3b90165a77d95cdf_root_uav_fill() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../shaderdump-hyphen/compute-3b90165a77d95cdf.cso");
    let original = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let container = Container::parse(&original).unwrap();
    let native = crate::native_programs(container.part(FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|p| p.stage == HwStage::Cs).unwrap();
    let StageIo::Cs(cs) = &native.io else { panic!("not compute") };
    assert_eq!(cs.threads, [32, 1, 1]);
    assert_eq!(cs.system_sgprs().0[0], Some(4));
    assert!(crate::spill::is_spilled(&native));
    let shader = super::translate(&original).expect("translate original UAV fill");
    assert_eq!(shader.root_uavs, [1]);
    assert!(shader.root_cbvs.is_empty() && shader.root_srvs.is_empty());
    for (slot, kind, word) in [(1, RootType::Descriptor, 0), (2, RootType::Constants, 1), (2, RootType::Constants, 2)] {
        assert!(shader.user_data.iter().any(|u| u.slot == slot && u.root == kind && u.dword == word), "missing binding {slot}:{word}");
    }
    // 0x28: index = group_x * 32 + local_x; 0x3c: index < s9.
    // 0x30 loads slot 2, so s9 is word 1 and s10 is word 2.
    // 0x44 loads slot 1; 0x4c and 0x74 store 4 copies of s10 at index*16.
    // The other cases exercise exact host bounds, not guessed native OOB values.
    let initial: Vec<u32> = (0..80*4).map(|i| 0x5a00_0000 | i).collect();
    let Ok(host) = Host::new(false) else { eprintln!("no hardware SM 6.6 device; skipped"); return; };
    let (root, pso) = host.compute_pipeline(&shader).expect("original fill PSO");
    for (count, groups, size, value) in [
        (0u32, 2u32, 1280u32, 0xdead_beefu32),
        (1, 1, 1280, 0),
        (31, 1, 1280, 0x1234_5678),
        (32, 2, 1280, u32::MAX),
        (33, 2, 1280, 0x89ab_cdef),
        (65, 2, 1280, 0x0102_0304),
        (64, 2, 33*16, 0x8765_4321),
        // A truncated 16-byte record must remain entirely untouched.
        (64, 2, 33*16+12, 0x8765_4321),
        // No complete record fits; every initialized sentinel must survive.
        (64, 2, 0, 0x8765_4321),
        (64, 2, 1, 0x8765_4321),
        (64, 2, 15, 0x8765_4321),
    ] {
        let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.slot, u.root, u.dword) {
                (1, RootType::Descriptor, 0) => size,
                (2, RootType::Constants, 1) => count,
                (2, RootType::Constants, 2) => value,
                other => panic!("unexpected binding {other:?}"),
            };
        }
        let output = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(ud)), (1, Root::Uav(initial.clone()))],
            views: &[], samplers: &[], groups: [groups, 1, 1],
        }).expect("original fill dispatch");
        let mut expected = initial.clone();
        let n = count.min(groups*32).min(size/16) as usize;
        expected[..n*4].fill(value);
        assert_eq!(output, vec![expected], "count={count}, groups={groups}, bytes={size}");
    }
}
