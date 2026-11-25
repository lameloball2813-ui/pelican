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
    let path = std::path::PathBuf::from(dir).join("pixel-7bdefdfde461f6e3.cso");
    let bytes = std::fs::read(path).expect("external cube-gradient pixel fixture");
    let c = Container::parse(&bytes).unwrap();
    let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
        .unwrap().into_iter().find(|n| n.stage == HwStage::Ps).unwrap();
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let gradients: Vec<_> = p.blocks.iter().flat_map(|bb| &bb.insts)
        .filter(|i| i.name().starts_with("image_sample_d"))
        .map(|i| i.machine.offset).collect();
    assert!(gradients.contains(&0x2abc), "original cube gradient location changed: {gradients:x?}");
    assert!(gradients.len() >= 3, "expected cube-gradient sampling sites: {gradients:x?}");
    let shader = super::translate(&bytes).expect("cube gradients should translate");
    let mut dxil = shader.dxil;
    super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
        .expect("DXIL validator")
        .validate(&mut dxil)
        .expect("original scene cube-gradient DXIL validation");
    // Pipeline creation on the actual default GPU is a separate requirement
    // from validation: root-resource declarations and driver acceptance must
    // agree. This doesn't purport to compare native texture filtering.
    let host = super::host::Host::new(false).expect("NVIDIA SM6.6 device");
    host.root_signature_for(&[(&super::translate(&bytes).unwrap(),
        windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL)])
        .expect("original pixel shader root signature");
}
