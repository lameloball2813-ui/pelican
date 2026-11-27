    use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn half_bias_sample_requires_exact_native_two_word_layout() {
        let supported = super::certified_half_bias_2d;
        assert!(supported(true, "image_sample_b", 1, false, 2));
        for (a16, name, dim, unorm, words) in [
            (false, "image_sample_b", 1, false, 2),
            (true, "image_sample_b_cl", 1, false, 2),
            (true, "image_sample_b_o", 1, false, 2),
            (true, "image_sample_b", 0, false, 2),
            (true, "image_sample_b", 4, false, 2),
            (true, "image_sample_b", 1, true, 2),
            (true, "image_sample_b", 1, false, 1),
            (true, "image_sample_b", 1, false, 3),
        ] {
            assert!(!supported(a16, name, dim, unorm, words),
                "unsupported sampling layout must not infer packed UV offsets");
        }
    }

    #[test]
    fn mesh_rendezvous_requires_every_path_exactly_once() {
        use super::{once_only_rendezvous, LaneExit, Bool};
        let branch = |taken, not_taken| LaneExit::Branch { when: Bool::TRUE, taken, not_taken };
        use LaneExit::{Jump, End, Indirect};
        assert!(once_only_rendezvous(&[branch(1, 2), Jump(3), Jump(3), End], 3));
        assert!(!once_only_rendezvous(&[branch(1, 2), Jump(3), End, End], 3));
        assert!(!once_only_rendezvous(&[branch(1, 2), Jump(3), Indirect, End], 3));
        assert!(!once_only_rendezvous(&[branch(1, 2), branch(1, 3), Jump(3), End], 3));
        assert!(!once_only_rendezvous(&[Jump(1), branch(2, 3), Jump(1), End], 1));
        assert!(!once_only_rendezvous(&[End, End], 1));
    }

    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        use super::super::container::SigElement;
        use super::half_input_partners;
        use crate::native::{PsInput, Usage};
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: true,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        assert_eq!(p.get(&5), None);
    }

    #[test]
    fn explicit_packed_input_tail_is_padding() {
        use super::super::container::SigElement;
        use super::packed_input_half;
        let element = |row, cols| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: true,
        };
        // Four halves in row 6 followed by only two in its explicit partner
        // row 7. The next two reads address the packing's zero padding.
        let sig = vec![element(6, 4), element(7, 2)];
        assert_eq!(
            packed_input_half(&sig, 6, Some(7), 5, true),
            Ok(Some((7, 1)))
        );
        assert_eq!(packed_input_half(&sig, 6, Some(7), 6, true), Ok(None));
        assert_eq!(packed_input_half(&sig, 6, Some(7), 7, true), Ok(None));
        // With an inferred partner the same gap is not assumed to be zero.
        assert!(packed_input_half(&sig, 6, Some(7), 6, false).is_err());
        assert!(packed_input_half(&sig, 6, None, 4, false).is_err());
    }

    /// Lowers `words` as a `stage` program (`wave64`: the console's wave
    /// size) with no resources or inputs.
    fn lower_program(
        words: &[u32],
        stage: HwStage,
        wave64: bool,
        io: StageIo,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io,
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
    }

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    /// A wave32 graphics program's host wave may have 64 lanes, whose upper
    /// half is a second console wave: a lane mask read as a number picks the
    /// reader's half of the ballot, and `v_mbcnt_lo` reports its rank as a
    /// host lane number.
    #[test]
    fn a_wave32_mask_read_takes_its_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
            0x7E08_0204, // v_mov_b32 v4, s4
            0xD765_0005,
            0x0001_006A, // v_mbcnt_lo_u32_b32 v5, vcc_lo, 0
            0xD766_0006,
            0x0002_0A6A, // v_mbcnt_hi_u32_b32 v6, vcc_lo, v5
            0xD765_0007,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v7, -1, 0
            0xBF85_0001, // s_cbranch_scc1 -> the export
            S_ENDPGM,    // no lane had its bit: done
            0xF800_180F,
            0x0706_0504, // exp mrt0, v4, v5, v6, v7 done vm
            S_ENDPGM,
        ];
        let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
        let f = |name| lowered.op_names.iter().position(|n| n == name);
        let (lane, ballot, bits) = (
            f("dx.op.waveGetLaneIndex").expect("no lane index for the pick"),
            f("dx.op.waveActiveBallot").expect("no ballot for the mask read"),
            f("dx.op.unaryBits.i32").expect("no bit count"),
        );
        let l = &lowered.insts;
        let lanes: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, i)| **i == format!("call f{lane} [k111]"))
            .map(|(i, _)| i)
            .collect();
        assert!(!lanes.is_empty(), "no lane index for the pick");
        let i_ballot = l
            .iter()
            .position(|i| i.starts_with(&format!("call f{ballot} [k116, ")))
            .expect("no ballot");
        let (i_x, i_y) = (
            at(l, &format!("extract i{i_ballot} 0")),
            at(l, &format!("extract i{i_ballot} 1")),
        );
        // The pick: upper host lanes read the ballot's second dword.
        let i_cmp = l
            .iter()
            .position(|i| {
                i.starts_with("cmp:") && lanes.iter().any(|n| i.ends_with(&format!("i{n} k32")))
            })
            .expect("no `lane >= 32` compare");
        at(l, &format!("select i{i_cmp} i{i_y} i{i_x}"));
        // v_mbcnt_lo counts below the lane in its own half; a wave32 mask's
        // high half is empty, so v_mbcnt_hi's count is of no bits.
        assert!(
            lanes
                .iter()
                .any(|n| l.contains(&format!("bin:And i{n} k31"))),
            "no `lane & 31` in {l:?}"
        );
        at(l, &format!("call f{bits} [k31, k0]"));
        // The count is a lane number, so a second wave's rank takes its
        // 32-lane base: `v_mbcnt_lo(-1)` is the host lane index.
        let bases: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.starts_with("bin:And ")
                    && lanes.iter().any(|n| **s == format!("bin:And i{n} k32"))
            })
            .map(|(i, _)| i)
            .collect();
        assert!(!bases.is_empty(), "no `lane & 32` wave base in {l:?}");
        for base in &bases {
            assert!(
                l.iter().any(|s| {
                    s.starts_with("bin:Add ")
                        && s["bin:Add ".len()..]
                            .split(' ')
                            .any(|a| a == format!("i{base}"))
                }),
                "the wave base `i{base}` is never added in {l:?}"
            );
        }
    }

    /// On a 64-lane host wave a wave32 program's lane-indexed ops use the
    /// lane's own console wave: a constant lane operand takes the wave's
    /// 32-lane base, and the first lane comes from its half's ballot.
    #[test]
    fn a_wave32_lane_op_uses_its_own_console_wave() {
        let words = [
            0xD760_0004,
            0x0001_0B02, // v_readlane_b32 v4, v2, 5
            0xD761_0005,
            0x0001_0A87, // v_writelane_b32 v5, 7, 5
            0x7E0C_0502, // v_readfirstlane_b32 v6, v2
            0x360E_0B04, // v_and_b32 v7, v4, v5
            0x360E_0D07, // v_and_b32 v7, v7, v6
            0x7D8A_0E80, // v_cmp_ne_u32 vcc, 0, v7
            0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
            0xBF85_0001, // s_cbranch_scc1 -> the export
            S_ENDPGM,
            0xF800_180F,
            0x0707_0707, // exp mrt0, v7, v7, v7, v7 done vm
            S_ENDPGM,
        ];
        let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
        let l = &lowered.insts;
        let lane_calls: Vec<usize> = l
            .iter()
            .enumerate()
            .filter(|(_, i)| i.starts_with("call f") && i.ends_with("[k111]"))
            .map(|(i, _)| i)
            .collect();
        assert!(!lane_calls.is_empty(), "no lane index in {l:?}");
        let bases: Vec<usize> = lane_calls
            .iter()
            .flat_map(|n| {
                l.iter()
                    .enumerate()
                    .filter(|(_, i)| **i == format!("bin:And i{n} k32"))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            })
            .collect();
        let used = |v: usize| -> bool {
            l.iter()
                .any(|i| i.ends_with(&format!("i{v}]")) || i.ends_with(&format!(" i{v}")))
        };
        // Console lane 5 of the lane's own wave is host lane 5 | (lane & 32),
        // for both v_readlane (readLaneAt) and v_writelane (lane equality).
        let ors: Vec<usize> = bases
            .iter()
            .filter_map(|b| l.iter().position(|i| i == &format!("bin:Or k5 i{b}")))
            .collect();
        assert_eq!(ors, Vec::from([2, 6]), "no `5 | base` host lanes in {l:?}");
        for or in &ors {
            assert!(used(*or), "the host lane `i{or}` is never used in {l:?}");
        }
        assert!(
            l.iter()
                .any(|i| i.starts_with("call f") && i.contains("k117, ")),
            "no waveReadLaneAt in {l:?}"
        );
        // v_readfirstlane takes the low set bit of the lane's own half of
        // the exec ballot, renumbered to a host lane.
        let ballot = l
            .iter()
            .position(|i| i.starts_with("call f") && i.contains("[k116, "))
            .expect("no exec ballot");
        let (x, y) = (
            at(l, &format!("extract i{ballot} 0")),
            at(l, &format!("extract i{ballot} 1")),
        );
        let own = l
            .iter()
            .position(|i| i.starts_with("select ") && i.ends_with(&format!("i{y} i{x}")))
            .expect("no own-half pick");
        let first = l
            .iter()
            .position(|i| i.starts_with("call f") && i.contains(&format!("k32, i{own}")))
            .expect("no first bit of the own half");
        let host_first = bases
            .iter()
            .find_map(|b| l.iter().position(|i| i == &format!("bin:Or i{first} i{b}")))
            .expect("the first lane is not renumbered");
        assert!(used(host_first), "no read at the first host lane in {l:?}");

        // On a wave64 program console lanes are host lanes one to one: the
        // same reads need no renumbering at all.
        let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
        let l = &l64.insts;
        assert!(
            !l.iter().any(|i| i.starts_with("bin:Or ")),
            "a wave64 lane operand is renumbered in {l:?}"
        );
        assert!(
            l.iter()
                .any(|i| i.starts_with("call f") && i.contains("k117, ") && i.contains("k5]")),
            "no direct waveReadLaneAt of lane 5 in {l:?}"
        );
        assert!(
            l.iter()
                .any(|i| i.starts_with("call f") && i.contains("k118, ")),
            "a wave64 first lane is not waveReadLaneFirst in {l:?}"
        );
    }

    /// The ballot-half pick is for wave32 graphics programs only: a wave64
    /// program's mask dwords are its own, and the console path (compute) is
    /// the whole host wave already, so neither reads the lane index for it.
    #[test]
    fn only_host_wave32_reads_take_a_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0x7E08_0204, // v_mov_b32 v4, s4
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            S_ENDPGM,
        ];
        let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
        assert!(!l64.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v0
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0xBF07_8004, // s_cmp_lg_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE85_0380, // s_mov_b32 s5, 0
            S_ENDPGM,
        ];
        let cs = lower_program(&words, HwStage::Cs, false, compute_io()).unwrap();
        assert!(!cs.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
    }

    fn pixel_io() -> StageIo {
        StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        }
    }

    fn compute_io() -> StageIo {
        StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [1, 1, 1],
        })
    }

    /// The index of the instruction `pat` names in `lines`.
    fn at(lines: &[String], pat: &str) -> usize {
        lines
            .iter()
            .position(|l| l == pat)
            .unwrap_or_else(|| panic!("no `{pat}` in {lines:?}"))
    }

    #[test]
    fn m0_relative_moves_with_two_indices_translate() {
        // v1 = 5, v2 = 7, M0 = source offset 0, destination offset 1, then
        // the M0-relative instruction and an LDS store of its v1 (an effect,
        // so a value the emitter cannot produce fails the translation).
        let program = |rel: u32| {
            [
                0x7E02_0285, // v_mov_b32 v1, 5
                0x7E04_0287, // v_mov_b32 v2, 7
                0x7E06_0280, // v_mov_b32 v3, 0
                0xBEFC_03FF, // s_mov_b32 m0, literal
                0x0001_0000,
                rel,
                0xD834_0000, // ds_write_b32 v3, v1
                0x0000_0103,
                S_ENDPGM,
            ]
        };
        for rel in [
            0x7E02_D101, // v_swaprel_b32 v1, v1
            0x7E02_9101, // v_movrelsd_2_b32 v1, v1
        ] {
            lowers(&program(rel), 1 << 15).unwrap_or_else(|e| panic!("{rel:08x}: {e}"));
        }
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "v_perm_b32")
        );

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();

        let container = crate::Container::parse(&emitted.dxil).unwrap();

        let features = container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }
    #[test]
    fn image_atomics_64_bit_lower_with_typed_int64_flag() {
        use crate::native::{RootParam, RootType};

        let run = |mimg_w0: u32| {
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0281,
                0x7E06_0280,
                0x7E08_0282,
                0x7E0A_0280,
                mimg_w0,
                0x0001_0200,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 8,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 8).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
            })
        };

        let add = run(0xF044_1388).unwrap();
        let cmp = run(0xF040_1F88).unwrap();
        for lowered in [add, cmp] {
            let sfi0 = crate::dxbc::Container::parse(&lowered.dxil)
                .unwrap()
                .part(crate::dxbc::FourCC::SFI0)
                .unwrap()
                .data;
            let features = u64::from_le_bytes(sfi0.try_into().unwrap());
            assert_ne!(features & 0x8000, 0, "INT64 feature bit missing");
            assert_ne!(
                features & 0x40_0000,
                0,
                "ATOMIC_INT64_ON_TYPED_RESOURCE feature bit missing"
            );
            assert_ne!(
                features & 0x1000_0000,
                0,
                "ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE feature bit missing"
            );
        }
    }

    #[test]
    fn sdwa_16bit_reserved_dst_unused_is_refused() {
        for dst_sel in [4u32, 5] {
            // v_cvt_i16_f16_sdwa v1, v0 dst_sel:{WORD_0,WORD_1} dst_unused:3 (reserved);
            // ds_write_b32 v0, v1 to keep v1 live.
            let sdwa_ext = (dst_sel << 8) | (3 << 11) | (4 << 16);
            let words = [
                0x7E02_A6F9, // v_cvt_i16_f16 v1, sdwa
                sdwa_ext,
                0xD834_0000,
                0x0000_0100, // ds_write_b32 v0, v1
                S_ENDPGM,
            ];
            let err = lowers(&words, 1 << 15).expect_err("reserved dst_unused must fail");
            assert!(
                matches!(
                    err,
                    crate::ShaderError::Unsupported {
                        what: "an SDWA destination of a 16-bit operation",
                        ..
                    }
                ),
                "unexpected error for dst_sel={dst_sel}: {err}"
            );
        }
    }

    #[test]
    fn image_sample_d_cl_and_gather4_l_translate() {
        use crate::native::{RootParam, RootType};
        let param = |root: RootType, user_data: u8, sgprs: u32| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        };
        let layout = [param(RootType::Table, 8, 1), param(RootType::Table, 9, 1)];

        // 1. image_sample_d_cl v[0:3], v[2:8], s[0:7], s[8:11]
        let sample_d_cl_words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            0xF08C_0F08,
            0x0040_0002, // image_sample_d_cl v[0:3], v[2:8], s[0:7], s[8:11]
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: layout.to_vec(),
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: sample_d_cl_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();

        // 2. image_gather4_l v[0:3], v[2:4], s[0:7], s[8:11]
        let gather4_l_words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            0xF110_0108,
            0x0040_0002, // image_gather4_l v[0:3], v[2:4], s[0:7], s[8:11]
            S_ENDPGM,
        ];
        let native_cs = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: layout.to_vec(),
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: gather4_l_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        };
        let p = ir::build(&native_cs.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native_cs.layout, native_cs.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native_cs).unwrap();
        lower(&Parts {
            native: &native_cs,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();
    }

    #[test]
    fn s_movrel_lowering() {
        // 1. s_movrels_b32 with static M0
        let static_rels = [
            0xBE84_03FF, // s_mov_b32 s4, 42
            42,
            0xBE85_03FF, // s_mov_b32 s5, 99
            99,
            0xBEFC_0381, // s_mov_b32 m0, 1
            0xBE80_2E04, // s_movrels_b32 s0, s4 (reads s[4 + 1] = s5)
            0xBE86_0300, // s_mov_b32 s6, s0
            S_ENDPGM,
        ];
        lowers(&static_rels, 0).unwrap();

        // 2. s_movrels_b32 with dynamic M0
        let dynamic_rels = [
            0xBE84_03FF, // s_mov_b32 s4, 42
            42,
            0xBE81_0381, // s_mov_b32 s1, 1
            0x80FC_8101, // s_add_u32 m0, s1, 1
            0xBE80_2E04, // s_movrels_b32 s0, s4 (reads s[4 + m0])
            0xBE86_0300, // s_mov_b32 s6, s0
            S_ENDPGM,
        ];
        lowers(&dynamic_rels, 0).unwrap();

        // 3. s_movreld_b32 with static M0 followed by readback
        let static_reld = [
            0xBE81_03FF, // s_mov_b32 s1, 77
            77,
            0xBEFC_0382, // s_mov_b32 m0, 2
            0xBE84_3001, // s_movreld_b32 s4, s1 (writes s[4 + 2] = s6)
            0xBE80_0306, // s_mov_b32 s0, s6
            S_ENDPGM,
        ];
        lowers(&static_reld, 0).unwrap();

        // 4. s_movreld_b32 with dynamic M0 followed by readback
        let dynamic_reld = [
            0xBE81_03FF, // s_mov_b32 s1, 88
            88,
            0xBE82_0382, // s_mov_b32 s2, 2
            0x80FC_8002, // s_add_u32 m0, s2, 0
            0xBE84_3001, // s_movreld_b32 s4, s1 (writes s[4 + m0])
            0xBE80_0306, // s_mov_b32 s0, s6
            S_ENDPGM,
        ];
        lowers(&dynamic_reld, 0).unwrap();

        // 5. Out-of-bounds M0 offsets (static & dynamic) clamp cleanly to 0 / no-op
        let oob_rels = [
            0xBE84_03FF, // s_mov_b32 s4, 42
            42,
            0xBEFC_03FF, // s_mov_b32 m0, 200 (out of SGPR bounds)
            200,
            0xBE80_2E04, // s_movrels_b32 s0, s4 (clamped to 0)
            0xBE86_0300, // s_mov_b32 s6, s0
            S_ENDPGM,
        ];
        lowers(&oob_rels, 0).unwrap();
    }

    #[test]
    fn cube_gradients_match_affine_face_directions() {
        // Independently differentiate the same native s,t,face convention
        // used for ordinary cube samples; never treat the face index as a
        // third derivative. Cover all faces and asymmetric signed gradients.
        fn direction(s: f32, t: f32, face: usize) -> [f32; 3] {
            let u = 2.0 * s - 3.0;
            let v = 2.0 * t - 3.0;
            match face {
                0 => [1.0, -v, -u],
                1 => [-1.0, -v, u],
                2 => [u, 1.0, v],
                3 => [u, -1.0, -v],
                4 => [u, -v, 1.0],
                5 => [-u, -v, -1.0],
                _ => unreachable!(),
            }
        }
        for face in 0..6 {
            for (s, t, ds, dt) in [
                (1.25, 1.75, 0.125, -0.0625),
                (1.5, 1.5, -0.25, 0.03125),
                (1.01, 1.98, 0.0, 0.125),
                (1.99, 1.02, 0.0625, 0.0),
            ] {
                let a = direction(s, t, face);
                let b = direction(s + ds, t + dt, face);
                let terms = [0.0, 2.0 * ds, -2.0 * ds, 2.0 * dt, -2.0 * dt];
                let got = super::CUBE_GRADIENT_TERMS[face].map(|i| terms[i as usize]);
                for axis in 0..3 {
                    assert!(
                        (got[axis] - (b[axis] - a[axis])).abs() <= 1.0e-5,
                        "face {face} axis {axis}: expected {}, got {}",
                        b[axis] - a[axis], got[axis],
                    );
                }
            }
        }
    }
