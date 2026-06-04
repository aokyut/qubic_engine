use cust::memory::{CopyDestination, DeviceBuffer, DeviceSlice};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use crate::board::uboard::*;
use crate::board::GetAction;
use std::time::Instant;

const SELECT_CUBIN:   &[u8] = include_bytes!("../../kernels/select.cubin");
const EXPAND_CUBIN:   &[u8] = include_bytes!("../../kernels/expand.cubin");
const SIMULATE_CUBIN: &[u8] = include_bytes!("../../kernels/simulate.cubin");
const BACKPROP_CUBIN: &[u8] = include_bytes!("../../kernels/backprop.cubin");

const MAX_NODES:  usize = 10_000_000;
const MAX_DEPTH:  usize = 256;
const N_WARPS:    u32   = 768;
const BLOCK_SIZE: u32   = 256;

pub struct GpuMcts {
    pub max_time:  u128,
    pub expand_th: u32,
    pub explore_c: f32,
}

impl GpuMcts {
    pub fn new(max_time: u128, expand_th: u32) -> Self {
        GpuMcts { max_time, expand_th, explore_c: 1.414 }
    }

    /// (action, na, val) のベクタを返す。
    /// action : 合法手ビット, na : 訪問数, val : root 側視点の価値 [0,1]
    pub fn run(&self, board: UBoard) -> Vec<(Action, u32, f32)> {
        self.run_inner(board).expect("GPU MCTS failed")
    }

    fn run_inner(&self, board: UBoard) -> Result<Vec<(Action, u32, f32)>, Box<dyn std::error::Error>> {
        let (att, def) = board;

        // ── CUDA コンテキスト初期化 ─────────────────────────────
        let _ctx = cust::quick_init()?;

        // ── PTX モジュールロード + 関数取得 ────────────────────
        let sel_mod = Module::from_cubin(SELECT_CUBIN,   &[])?;
        let exp_mod = Module::from_cubin(EXPAND_CUBIN,   &[])?;
        let sim_mod = Module::from_cubin(SIMULATE_CUBIN, &[])?;
        let bp_mod  = Module::from_cubin(BACKPROP_CUBIN, &[])?;

        let k_select   = sel_mod.get_function("mcts_select").unwrap();
        let k_expand   = exp_mod.get_function("mcts_expand").unwrap();
        let k_simulate = sim_mod.get_function("mcts_simulate").unwrap();
        let k_backprop = bp_mod.get_function("mcts_backprop").unwrap();

        // ── GPU バッファ確保 ────────────────────────────────────
        // ルートノードの att/def を含む初期化ベクタ (一時的 CPU 確保)
        let (d_att, d_def) = {
            let mut ha = vec![0u64; MAX_NODES];
            let mut hd = vec![0u64; MAX_NODES];
            ha[0] = att;
            hd[0] = def;
            let da = DeviceBuffer::from_slice(&ha).unwrap();
            let dd = DeviceBuffer::from_slice(&hd).unwrap();
            (da, dd)
        };

        
        // 残りのノードフィールドはゼロ初期化で OK
        // (total_val=0.0, first_child=0, n_children=0, vl=0, terminal=0)
        let mut d_visits      = DeviceBuffer::<u32>::zeroed(MAX_NODES)?;
        // root.visits を expand_th に設定: expand 条件 visits>=expand_th を初回から満たす。
        // backprop.ptx が毎シミュレーション visits[0] を +1 するため、
        // 以降は expand_th + シミュレーション数 として成長する。
        {
            let root_visits = self.expand_th.max(1);
            let mut s = d_visits.index(0..1);
            s.copy_from(&[root_visits])?;
            // 書き込み確認
            let v: Vec<u32> = d_visits.index(0..1).as_host_vec().unwrap();
            // eprintln!("[DEBUG] root.visits={}, expand_th={}", v[0], self.expand_th);
        }
        let     d_total_val   = DeviceBuffer::<f32>::zeroed(MAX_NODES)?;
        let     d_first_child = DeviceBuffer::<u32>::zeroed(MAX_NODES)?;
        let     d_n_children  = DeviceBuffer::<u32>::zeroed(MAX_NODES)?;
        let     d_parent      = DeviceBuffer::<u32>::zeroed(MAX_NODES)?;
        let     d_vl          = DeviceBuffer::<i32>::zeroed(MAX_NODES)?;
        let     d_terminal    = DeviceBuffer::<u8>::zeroed(MAX_NODES)?;
        
        // アトミックカウンタ: ルートは 0 番, 次の空きは 1
        // from_slice の代わりに zeroed + copy_from で確実に初期化する
        let mut d_node_count = DeviceBuffer::<u32>::zeroed(1)?;
        {
            let mut s = d_node_count.index(0..1);
            s.copy_from(&[1u32])?;
            let v: Vec<u32> = d_node_count.index(0..1).as_host_vec().unwrap();
            // eprintln!("[DEBUG] node_count init={}", v[0]);
        }
        
        // 一時バッファ (N_WARPS 本の探索路)
        let d_leaf_out    = DeviceBuffer::<u32>::zeroed(N_WARPS as usize)?;
        let d_path_out    = DeviceBuffer::<u32>::zeroed(N_WARPS as usize * MAX_DEPTH)?;
        let d_path_len    = DeviceBuffer::<u32>::zeroed(N_WARPS as usize)?;
        let d_sim_results = DeviceBuffer::<f32>::zeroed(N_WARPS as usize)?;
        
        // ── ストリーム + グリッド設定 ────────────────────────────
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        let grid_select: u32 = (N_WARPS * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let grid_other:  u32 = (N_WARPS       + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // ── メインループ ────────────────────────────────────────
        let mut seed: u64 = 0x6c62272e07bb0142;
        let start = Instant::now();
        
        while start.elapsed().as_millis() < self.max_time {
            seed = seed.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

            seed ^= (att.wrapping_mul(0x9e3779b97f4a7c15))
                ^ (def.wrapping_mul(0x6c62272e07bb0142));
        
            macro_rules! sync_check {
                ($label:expr) => {
                    stream.synchronize().unwrap_or_else(|e| panic!("{} failed: {:?}", $label, e));
                };
            }

            unsafe {
                cust::launch!(k_select<<<grid_select, BLOCK_SIZE, 0, stream>>>(
                    d_visits.as_device_ptr(), d_total_val.as_device_ptr(),
                    d_first_child.as_device_ptr(), d_n_children.as_device_ptr(),
                    d_parent.as_device_ptr(), d_vl.as_device_ptr(),
                    d_leaf_out.as_device_ptr(), d_path_out.as_device_ptr(),
                    d_path_len.as_device_ptr(), N_WARPS, self.explore_c
                )).unwrap();
            }
            sync_check!("SELECT");

            unsafe {
                cust::launch!(k_expand<<<grid_other, BLOCK_SIZE, 0, stream>>>(
                    d_att.as_device_ptr(), d_def.as_device_ptr(),
                    d_visits.as_device_ptr(), d_total_val.as_device_ptr(),
                    d_first_child.as_device_ptr(), d_n_children.as_device_ptr(),
                    d_parent.as_device_ptr(), d_vl.as_device_ptr(),
                    d_terminal.as_device_ptr(), d_leaf_out.as_device_ptr(),
                    d_node_count.as_device_ptr(), N_WARPS, self.expand_th
                )).unwrap();
            }
            sync_check!("EXPAND");

            unsafe {
                cust::launch!(k_simulate<<<grid_other, BLOCK_SIZE, 0, stream>>>(
                    d_att.as_device_ptr(), d_def.as_device_ptr(),
                    d_terminal.as_device_ptr(), d_leaf_out.as_device_ptr(),
                    d_sim_results.as_device_ptr(), N_WARPS, seed
                )).unwrap();
            }
            sync_check!("SIMULATE");

            unsafe {
                cust::launch!(k_backprop<<<grid_other, BLOCK_SIZE, 0, stream>>>(
                    d_visits.as_device_ptr(), d_total_val.as_device_ptr(),
                    d_vl.as_device_ptr(), d_path_out.as_device_ptr(),
                    d_path_len.as_device_ptr(), d_sim_results.as_device_ptr(),
                    N_WARPS
                )).unwrap();
            }
            sync_check!("BACKPROP");
        }
        println!("hoge");
    
        // ── 全子ノードの統計を収集 ──────────────────────────────
        let h_fc: Vec<u32> = d_first_child.index(0..1).as_host_vec().unwrap();
        let h_nc: Vec<u32> = d_n_children.index(0..1).as_host_vec().unwrap();

        let fc = h_fc[0] as usize;
        let nc = h_nc[0] as usize;
        let nc_cnt: Vec<u32> = d_node_count.index(0..1).as_host_vec().unwrap();
        // eprintln!("[DEBUG] fc={fc}, nc={nc}, node_count={}", nc_cnt[0]);

        if nc == 0 {
            // 展開前 (時間切れ等): 合法手を enumerte して全て 0 で返す
            let mut result = Vec::new();
            let mut mask = board.get_valid();
            while mask != 0 {
                let lsb = mask & mask.wrapping_neg();
                result.push((lsb, 0u32, 0.0f32));
                mask &= mask - 1;
            }
            return Ok(result);
        }


        // ルートの全子ノードの visits / total_val / def を一括転送
        let h_visits:    Vec<u32> = d_visits.index(fc..fc + nc).as_host_vec()?;
        let h_total_val: Vec<f32> = d_total_val.index(fc..fc + nc).as_host_vec()?;
        let h_def:       Vec<u64> = d_def.index(fc..fc + nc).as_host_vec()?;

        // (action, na, val) に変換
        // action = child.def ^ att  (child.def = att | action なので)
        // total_val は子手番プレイヤー(=相手)の勝率を蓄積しているため、
        // 自手番視点の勝率 = 1 - total_val/na
        let results: Vec<(Action, u32, f32)> = (0..nc)
            .map(|i| {
                let action = h_def[i] ^ att;
                let na     = h_visits[i];
                let val    = if na > 0 { 1.0 - h_total_val[i] / na as f32 } else { 0.0 };
                (action, na, val)
            })
            .collect();

        Ok(results)
    }
}


impl GetAction for GpuMcts{
    fn get_action(&self, b: &crate::board::Board) -> u8 {
        use std::time::Instant;
        let (att, def) = b.get_att_def();
        let t = Instant::now();
        let mut ans = self.run((att, def));
        let time = t.elapsed().as_micros();
        let mut max_action = 0;
        let mut max_val = 0;
        ans.sort_by(|a, b| b.1.cmp(&a.1));
        let mut sum_node = 0;
        if cfg!(feature="view"){
            for (action, na, q) in ans{
                let action = action.trailing_zeros() % 16;
                if na >= max_val{
                    max_val = na;
                    max_action = action as u8;
                }
                sum_node += na as u64;
                println!("[Action:{action}]na:{na}, q:{q}");
            }
            println!("total_node:{sum_node}, total_time:{time}[μs], nps:{}[nps]", sum_node * 1000_000 / (time as u64));
        }
        return max_action;
    }
}