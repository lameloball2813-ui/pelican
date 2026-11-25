//! Regression tests using original title shaders kept outside the repository.
//! Set `VICTORY_PIXEL_DESCRIPTOR_DIR` to the directory holding either
//! `pixel-<hash>.cso` or `pixel-one-<hash>/pixel-<hash>.cso`.

use super::{translate, validator};

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
