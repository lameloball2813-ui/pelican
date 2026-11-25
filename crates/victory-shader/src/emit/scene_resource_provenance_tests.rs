//! Original-scene resource-address provenance diagnostics. These external
//! fixtures remain outside the repository; neither shader bytes nor derived
//! code are committed. Never authorize a GPU access merely from a root
//! constant which happens to contain a title virtual address.
use crate::dxbc::{Container, FourCC};
use crate::native::HwStage;
use crate::ShaderError;

#[test]
fn original_scene_spilled_constant_pointers_stay_rejected() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let path = std::path::PathBuf::from(dir).join("compute-f6e47efa50226beb.cso");
    let bytes = std::fs::read(path).expect("original compute fixture outside repository");
    let c = Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|n| n.stage == HwStage::Cs).unwrap();
    assert!(crate::spill::is_spilled(&native));
    let (mut p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let rewritten = crate::spill::user_data(&mut p, &native);
    assert!(rewritten > 0, "real spilled arguments must remain supported");
    let st = if rewritten > 0 { crate::structure::structure(&p).unwrap() } else { st.unwrap() };
    let err = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap_err();
    assert_eq!(err, ShaderError::Unsupported {
        offset: 0x13ec,
        what: "an address in an SGPR that is no root parameter",
    }, "a pointer passed as root constants requires explicit address translation, not a fabricated root descriptor");
    // The native instruction at the refusal is a global load, not an
    // ordinary root descriptor read. Preserve both its machine offset and
    // the failure-closed provenance behavior across translator changes.
    let inst = p.blocks.iter().flat_map(|bb| &bb.insts)
        .find(|i| i.machine.offset == 0x13ec).expect("original address use");
    assert!(inst.name().starts_with("global_load_"), "{}", inst.name());
    // The original address arrives via the VGPR pair v8:v9, copied from
    // spilled slot 8 constants (not a separately bound root descriptor).
    // A GPU VA embedded in constants needs explicit title-to-host mapping.
    assert!(matches!(inst.fields, crate::gfx10::Fields::Flat {
        addr: Some(crate::gfx10::Operand::Vector { reg: 8, dwords: 2 }),
        saddr: Some(crate::gfx10::Operand::Scalar { code: crate::gfx10::NULL, .. }),
        offset: 0, ..
    }), "actual original indirect address structure changed: {:?}", inst.fields);
    assert!(native.layout.iter().any(|e| e.root == crate::native::RootType::Constants
        && e.slot == 8 && e.user_data == 32 && e.sgprs == 14),
        "original pointer-bearing root constants are slot 8 dwords 0..13");
}

/// A cube's original sample_d layout has two gradients per screen direction
/// followed by three face coordinates. The 3D DXIL gradients are obtained by
/// differentiating those face coordinates, never by reading nonexistent
/// eighth/ninth derivative registers.
#[test]
fn original_scene_cube_gradient_pixel_shader_validates() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    let mut validator = super::validator::Validator::load(
        super::validator::Validator::default_path().as_deref())
        .expect("DXIL validator");
    // Pipeline creation on the actual default GPU is a separate requirement
    // from validation: root-resource declarations and driver acceptance must
    // agree. This doesn't purport to compare native texture filtering.
    let host = super::host::Host::new(false).expect("NVIDIA SM6.6 device");
    for (hash, first) in [
        ("7bdefdfde461f6e3", 0x2abc),
        ("ee41ebc3eb3fbbdc", 0x39c4),
        ("2deec721a63fe8a5", 0x2824),
    ] {
        let path = dir.join(format!("pixel-{hash}.cso"));
        let bytes = std::fs::read(path).expect("external cube-gradient pixel fixture");
        let c = Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|n| n.stage == HwStage::Ps).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let gradients: Vec<_> = p.blocks.iter().flat_map(|bb| &bb.insts)
            .filter(|i| i.name().starts_with("image_sample_d"))
            .map(|i| i.machine.offset).collect();
        assert!(gradients.contains(&first),
            "{hash}: original cube gradient location changed: {gradients:x?}");
        assert!(gradients.len() >= 3, "{hash}: expected cube-gradient sampling sites");
        for inst in p.blocks.iter().flat_map(|bb| &bb.insts)
            .filter(|i| i.name().starts_with("image_sample_d")
                && matches!(i.fields, crate::gfx10::Fields::Image { dim: 3, .. })) {
            let crate::gfx10::Fields::Image { dim, a16, addr, .. } = &inst.fields else {
                unreachable!()
            };
            assert_eq!(*dim, 3, "{hash} {:#x}: cube dimension", inst.machine.offset);
            assert!(!*a16, "{hash} {:#x}: unexpected half-gradient format", inst.machine.offset);
            let words: usize = addr.iter().map(|op| match op {
                crate::gfx10::Operand::Vector { dwords, .. } => usize::from(*dwords),
                _ => panic!("{hash}: image gradient address not VGPR"),
            }).sum();
            assert!(matches!(words, 7 | 8), "{hash} {:#x}: unexpected {words} native address words", inst.machine.offset);
        }
        let shader = super::translate(&bytes)
            .unwrap_or_else(|e| panic!("{hash}: cube gradients should translate: {e}"));
        let mut dxil = shader.dxil.clone();
        validator.validate(&mut dxil)
            .unwrap_or_else(|e| panic!("{hash}: invalid cube-gradient DXIL: {e}"));
        host.root_signature_for(&[(&shader,
            windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL)])
            .unwrap_or_else(|e| panic!("{hash}: root signature: {e}"));
    }
}

/// Diagnostic for the original water/material 2D descriptor metadata path.
/// Record each scalar descriptor-load site and its exact SSA consumers before
/// changing production provenance checks. In particular, extracting width,
/// height and base mip from three *separate* loads must never be mistaken for
/// one descriptor of an unrelated heap slot.
#[test]
fn original_scene_a7a20_separate_dimension_loads_audit() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    let bytes = std::fs::read(std::path::PathBuf::from(dir)
        .join("pixel-a7a20f06a33260f9.cso")).unwrap();
    let c = Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|n| n.stage == HwStage::Ps).unwrap();
    let (p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let offset = |off: usize| p.blocks.iter().flat_map(|b| &b.insts)
        .find(|i| i.machine.offset == off).unwrap();
    let mut signatures = Vec::new();
    for off in [0x550, 0x590, 0x5c8] {
        let i = offset(off);
        assert_eq!(i.name(), "s_load_dwordx8");
        let site = p.blocks.iter().enumerate().find_map(|(b, bb)| bb.insts.iter()
            .position(|i| i.machine.offset == off).map(|j| (b,j))).unwrap();
        let (_, a) = res.scalar.iter().find(|(s,_)| *s == site).unwrap();
        let dynamic = a.dynamic.unwrap();
        // The three scalar first-lane reads are distinct SSA definitions of
        // the same VGPR input, under three separate EXEC snapshots.
        let origin = p.blocks.iter().flat_map(|bb| &bb.insts)
            .find(|w| w.writes.iter().any(|&(_, v)| v == dynamic))
            .expect("first-lane index for this descriptor load");
        assert_eq!(origin.name(), "v_readfirstlane_b32");
        assert_eq!(origin.read(crate::ir::Reg::V(18)), Some(crate::ir::Value(1065)),
            "all descriptor reads use the original per-lane index");
        // These three scalar metadata reads are NOT yet among the resource
        // tracker's certified descriptor waterfalls. Their shared VGPR
        // address alone does not prove lane equivalence after EXEC changes;
        // keep the original pipeline unsupported until that is proven.
        assert!(!res.waterfall.reads.contains(&dynamic));
        signatures.push((a.param, a.offset, a.dynamic));
    }
    // SSA differs because the native waterfall reloads the scalar index in
    // three blocks. Do not assume those values are identical from the root
    // parameter alone: the proving pass must inspect their defining chains.
    assert!(signatures.windows(2).all(|w| w[0].0 == w[1].0 && w[0].1 == w[1].1));
    assert!(signatures.windows(2).all(|w| w[0].2 != w[1].2));
    assert!(matches!(super::translate(&bytes), Err(ShaderError::Unsupported { .. })),
        "this audit must be retired or upgraded when the real original pixel shader is supported");
}

/// These original mesh programs lose visible geometry when their sampler
/// descriptor is assembled from computed per-lane VGPRs. Nearby native loads
/// alone do not certify their SSA origins or EXEC coverage. Preserve exact
/// failure sites until a full incoming-word/path proof is available.
#[test]
fn original_scene_mesh_sampler_provenance_remains_guarded() {
    let Some(dir) = std::env::var_os("VICTORY_MESH_DESCRIPTOR_DIR") else { return };
    for (hash, site) in [
        ("adc5c689b0552cc2", 0x5ba8usize),
        ("903776ba469c8bac", 0x57b4),
        ("89a622617919c92a", 0x5fe8),
        ("fb5e4cf7091f9525", 0x5b84),
    ] {
        let bytes = std::fs::read(std::path::PathBuf::from(&dir)
            .join(format!("mesh-{hash}.cso"))).expect("original missing scene mesh shader");
        let c = Container::parse(&bytes).unwrap();
        let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
            .unwrap().into_iter().find(|n| n.stage == HwStage::Ms).unwrap();
        let (p, structured) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs)
            .expect("real mesh SSA");
        let err = crate::resource::track(&p, &native.layout, native.stage, structured.as_ref().ok())
            .expect_err("unproven sampler descriptor must stay rejected");
        assert_eq!(err, ShaderError::Unsupported {
            offset: site,
            what: "a descriptor read from the first lane of computed VGPRs",
        }, "{hash}: do not recover geometry with an unproven sampler");
        let at = p.blocks.iter().flat_map(|bb| &bb.insts)
            .find(|i| i.machine.offset == site).expect("original sample offset");
        assert!(at.name().starts_with("image_sample"), "{hash}: {}", at.name());
        if hash == "adc5c689b0552cc2" {
            let instr = |offset| p.blocks.iter().flat_map(|bb| &bb.insts)
                .find(|i| i.machine.offset == offset).expect("original sampler instruction");
            let load_a = instr(0x5218);
            let load_b = instr(0x5994);
            assert_eq!((load_a.name(), load_b.name()), ("s_load_dwordx4", "s_load_dwordx4"));
            assert_eq!(load_a.fields, load_b.fields, "same native sampler offset");
            for reg in [24u8, 25] {
                assert_eq!(load_a.read(crate::ir::Reg::S(reg)),
                    load_b.read(crate::ir::Reg::S(reg)),
                    "same scalar root-pointer SSA, not merely matching SGPR numbers");
            }
            // The later four readfirstlane operands are *not* scalar copies
            // of these loads. They are phis at block 190, containing the
            // branch at block 183 and the alternative block 189. At block
            // 183, one incoming path still holds geometry calculations.
            // EXEC coverage, not identical load addresses, must prove which
            // lanes actually observe the sampler writes.
            for reg in 76u8..80 {
                let firstlane = instr(0x5b78 + usize::from(reg - 76) * 4);
                assert_eq!(firstlane.name(), "v_readfirstlane_b32");
                let value = firstlane.read(crate::ir::Reg::V(reg)).unwrap();
                let joined = p.blocks[190].phis.iter()
                    .find(|phi| phi.value == value && phi.reg == crate::ir::Reg::V(reg))
                    .expect("per-lane descriptor is still a control-flow phi");
                assert_eq!(joined.args.len(), 2);
            }
            let first = instr(0x5b78).read(crate::ir::Reg::V(76)).unwrap();
            let source = p.blocks[190].phis.iter().find(|phi| phi.value == first).unwrap();
            let earlier = source.args[0];
            let inner = p.blocks[183].phis.iter().find(|phi| phi.value == earlier).unwrap();
            let geometry = inner.args[0];
            let defining = p.blocks.iter().flat_map(|bb| &bb.insts)
                .find(|i| i.writes.iter().any(|&(_, v)| v == geometry)).unwrap();
            assert_eq!(defining.name(), "v_sub_f32",
                "unwritten lanes still carry geometry, not a sampler word");
        }
    }
}
