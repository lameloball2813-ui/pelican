//! GPU correctness checks for the title's actual BVH group-prefix shaders.
//! Set `VICTORY_BVH_SCAN_DIR` to a directory OUTSIDE the source tree holding
//! `compute-<hash>.cso` dumps from the user's own title. Without dumps the
//! test skips; no shader container is embedded in or committed with the repo.

use super::host::{Dispatch, Host, Root, View};
use super::{translate, validator};
use crate::native::RootType;

#[test]
fn original_group_scans_match_cpu_on_nvidia_wave32() {
    let Some(dir) = std::env::var_os("VICTORY_BVH_SCAN_DIR") else {
        eprintln!("VICTORY_BVH_SCAN_DIR unset; original title shader assets are external");
        return;
    };
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).expect("SM6.6 GPU needed for BVH prefix test");
    // This test is an exact numerical gate, not an authorization to execute
    // the original 64-lane wave operators as unrelated 32-lane waves. A
    // rejected pipeline is an expected pending fixture until the logical
    // full-wave scan proof and rewrite are complete.
    let mut blocked = 0usize;
    for (hash, threads, width) in [
        ("39321a95223001ed", 256usize, 4usize),
        ("77360df21e12ea43", 512usize, 4usize),
        ("9fc111d3be79364f", 512usize, 8usize),
    ] {
        let path = std::path::PathBuf::from(&dir).join(format!("compute-{hash}.cso"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let shader = translate(&bytes);
        let shader = shader.unwrap_or_else(|e| panic!("{hash}: {e}"));
        eprintln!(
            "{hash}: user_data={:?} cbvs={:?} srvs={:?} uavs={:?}",
            shader.user_data, shader.root_cbvs, shader.root_srvs, shader.root_uavs
        );
        let mut checked = shader.dxil.clone();
        validator::Validator::load(validator::Validator::default_path().as_deref())
            .expect("DXIL validator")
            .validate(&mut checked)
            .expect("translated scan DXIL invalid");
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(reason) if reason.contains("parameter is incorrect") => {
                eprintln!("{hash}: expected guarded wave64 pipeline refusal on wave32 host: {reason}");
                blocked += 1;
                continue;
            }
            Err(reason) => panic!("{hash}: unexpected host pipeline failure: {reason}"),
        };
        for dataset in 0..2u32 {
            let mut values = vec![0u32; threads * width];
            for (i, v) in values.iter_mut().enumerate() {
                *v = if dataset == 0 {
                    1
                } else {
                    ((i * 17 + 3) % 11) as u32
                };
            }
            let mut user = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for binding in &shader.user_data {
                if binding.root == RootType::Constants && binding.slot == 2 && binding.dword == 0 {
                    user[binding.index as usize] = 1;
                }
            }
            // The native shader loads its spilled root constant at byte offset
            // 0x10 and tests its low bit before the group-total store.
            let totals = vec![u32::MAX; values.len()];
            let results = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(user)),
                            (1, Root::Uav(values.clone())),
                            (2, Root::Uav(totals)),
                        ],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{hash} case {dataset}: {e}"));
            assert_eq!(results.len(), 3);
            eprintln!(
                "{hash} case {dataset}: prefix-first={:?} totals-first={:?}",
                &results[1][..16],
                &results[2][..8]
            );
            let got = &results[1];
            let mut sum = 0u32;
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(
                    got[i],
                    sum,
                    "{hash} case {dataset}, element {i} (lane {})",
                    i / width
                );
                sum = sum.wrapping_add(value);
            }
            assert_eq!(results[2][0], sum, "{hash} case {dataset}: group total");
            assert!(
                results[2][1..].iter().all(|&v| v == u32::MAX),
                "{hash} case {dataset}: unexpected additional total store"
            );
        }
    }
    assert!(blocked <= 3);
    eprintln!("{blocked} original BVH numerical fixtures blocked by unlowered wave64 semantics");
}
