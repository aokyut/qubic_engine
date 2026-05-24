use std::f64::consts::LN_10;
use crate::board::{Board, GetAction};

pub struct BayesElo {
    pub delta: f64,         // レーティング差
    pub draw_elo: f64,      // 引き分けやすさ
    pub delta_std: f64,     // Δ の標準偏差
    pub draw_elo_std: f64,  // d_e の標準偏差
    pub ci95: (f64, f64),   // Δ の95% CI
}

pub fn bayes_elo(w: u64, d: u64, l: u64) -> Option<BayesElo> {
    if w == 0 || l == 0 { return None; } // 退化ケース
    
    let (w, d, l) = (w as f64, d as f64, l as f64);
    let n = w + d + l;
    let c = 200.0; // 普通に書く
    
    // MLE 点推定
    let delta = c * (w * (w + d) / (l * (l + d))).log10();
    let draw_elo = c * ((w + d) * (l + d) / (w * l)).log10();
    
    // ∂Δ/∂W, ∂Δ/∂L （200/ln10 を係数として）
    let k = 200.0 / LN_10;
    let d_delta_dw =  k * (1.0 / w + 1.0 / (w + d));
    let d_delta_dl = -k * (1.0 / l + 1.0 / (l + d));
    
    // ∂d_e/∂W, ∂d_e/∂L
    let d_de_dw = -k * (1.0 / w - 1.0 / (l + d));  
    let d_de_dl = -k * (1.0 / l - 1.0 / (w + d));
    // 注：d_e の微分は ∂/∂W [ln(n-L) - ln W] 等から
    // 厳密には W,L を独立変数とし D = n-W-L として扱う
    // → ここは下で改めて多項モデルで処理
    
    // 多項分布の共分散
    let var_w = w * (n - w) / n;
    let var_l = l * (n - l) / n;
    let cov_wl = -w * l / n;
    
    // Δ の分散
    let var_delta = d_delta_dw.powi(2) * var_w
                  + d_delta_dl.powi(2) * var_l
                  + 2.0 * d_delta_dw * d_delta_dl * cov_wl;
    let delta_std = var_delta.sqrt();
    
    // d_e の分散も同様（W, D, L すべてで微分するのが正しい）
    let draw_elo_std = compute_draw_elo_std(w, d, l, n);
    
    let ci95 = (delta - 1.96 * delta_std, delta + 1.96 * delta_std);
    
    Some(BayesElo { delta, draw_elo, delta_std, draw_elo_std, ci95 })
}

/// d_e は (W, D, L) → f の偏微分で全項を使う
fn compute_draw_elo_std(w: f64, d: f64, l: f64, n: f64) -> f64 {
    let k = 200.0 / LN_10;
    // d_e = k * [ln(W+D) + ln(L+D) - ln W - ln L]
    let dw =  k * (1.0 / (w + d) - 1.0 / w);
    let dd =  k * (1.0 / (w + d) + 1.0 / (l + d));
    let dl =  k * (1.0 / (l + d) - 1.0 / l);
    
    // 多項共分散
    let cov = |i: f64, j: f64, same: bool| {
        if same { i * (n - i) / n } else { -i * j / n }
    };
    
    let var = dw.powi(2) * cov(w, w, true)
            + dd.powi(2) * cov(d, d, true)
            + dl.powi(2) * cov(l, l, true)
            + 2.0 * dw * dd * cov(w, d, false)
            + 2.0 * dw * dl * cov(w, l, false)
            + 2.0 * dd * dl * cov(d, l, false);
    var.sqrt()
}

// 100 * a / b % or ?
fn parcent_string(a:u64, b:u64) -> String{
    if b == 0{
        return String::from("?");
    }else{
        return format!("{}", 100 * a / b);
    }
}
fn elo2string(b: Option<BayesElo>) -> String{
    match b{
        Some(BayesElo { delta, draw_elo, delta_std, draw_elo_std, ci95 }) =>{
            format!("{delta:.0}±{delta_std:.1}(d:{draw_elo:.0})")
        },
        None => {
            format!("?")
        }
    }
}

pub fn bayes_elo_from_boards(bs: &[Board], tar_agent: &impl GetAction, test_agent: &impl GetAction, threshold: f64, render: bool) -> (BayesElo, f32, f32, f32){
    use crate::board::play_actor_from;
    use indicatif::{ProgressBar, ProgressStyle};
    let (a1, a2) = (tar_agent, test_agent);
    let pb = ProgressBar::new_spinner();

    if bs.len() == 0{
        return (BayesElo {delta:0.0, draw_elo:0.0, delta_std:0.0, draw_elo_std:0.0, ci95:(0.0, 0.0)}, 0.0, 0.0, 0.0);
    }
    let mut idx = 0;
    let (mut w, mut l, mut d) = (0, 0, 0);

    let mut now_elo = None;
    loop{
        pb.set_message(format!(
            "[Target VS Test {}]W/L/D:{w}/{l}/{d} | ELO: {}, Score: {}%",
            w + l + d,
            elo2string(now_elo),
            parcent_string(w, w + l + d),
        ));
        let (s1, s2) = play_actor_from(bs[idx].clone(), a1, a2, render);
        if s1 > s2{
            w += 1;
        }else if s2 > s1{
            l += 1;
        }else{
            d += 1;
        }

        now_elo = bayes_elo(w, d, l);
        pb.set_message(format!(
            "[Target VS Test {}]W/L/D:{w}/{l}/{d} | ELO: {}, Score: {}%",
            w + l + d,
            elo2string(now_elo),
            parcent_string(w, w + l + d),
        ));
        let (s2, s1) = play_actor_from(bs[idx].clone(), a2, a1, render);
        if s1 > s2{
            w += 1;
        }else if s2 > s1{
            l += 1;
        }else{
            d += 1;
        }

        now_elo = bayes_elo(w, d, l);
        if let Some(BayesElo { delta, draw_elo, delta_std, draw_elo_std, ci95 }) = now_elo{
            if delta_std < threshold{
                pb.finish();
                if render{
                    println!("[Finish bayes_elo_from_boards]ELO:{}", elo2string(now_elo));
                    return (BayesElo { delta, draw_elo, delta_std, draw_elo_std, ci95 }, w as f32, l as f32, d as f32);
                }
            }
        }
        idx = (idx + 1) % bs.len();
    }
}