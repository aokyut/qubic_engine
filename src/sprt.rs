use crate::board::{eval_actor, eval_actor_from_boards, n_step_board, Board, GetAction};
use crate::db::BoardDB;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::time;

/// Sequential Probability Ratio Test for efficient model comparison
/// 
/// SPRT allows early termination when statistical significance is reached,
/// minimizing the number of games needed for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPRT {
    /// Elo difference for null hypothesis (H0)
    pub elo0: f32,
    /// Elo difference for alternative hypothesis (H1)
    pub elo1: f32,
    /// Type I error probability (false positive)
    pub alpha: f32,
    /// Type II error probability (false negative)
    pub beta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SPRTResult {
    /// H0 accepted: models are similar (no significant difference)
    H0Accepted { games: usize, llr: f32 },
    /// H1 accepted: new model is significantly stronger
    H1Accepted { games: usize, llr: f32 },
    /// Continue testing: not enough evidence yet
    Continue { games: usize, llr: f32 },
    /// Maximum games reached without conclusion
    Inconclusive { games: usize, llr: f32 },
}

impl SPRTResult{
    pub fn get_games(&self) -> usize{
        use SPRTResult::*;
        match self{
            Continue { games, llr } =>{
                *games
            },
            H0Accepted { games, llr } => {
                *games
            },
            H1Accepted { games, llr } => {
                *games
            },
            Inconclusive { games, llr } => {
                *games
            }
        }
    }
}

impl SPRT {
    /// Create a new SPRT with default parameters
    /// 
    /// Default: detect Elo difference of 10 with α=β=0.05
    pub fn new() -> Self {
        SPRT {
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
        }
    }

    /// Create SPRT with custom Elo bounds
    pub fn with_elo_bounds(elo0: f32, elo1: f32) -> Self {
        SPRT {
            elo0,
            elo1,
            alpha: 0.05,
            beta: 0.05,
        }
    }

    /// Convert Elo difference to expected score
    fn elo_to_score(elo_diff: f32) -> f32 {
        1.0 / (1.0 + 10.0_f32.powf(-elo_diff / 400.0))
    }

    /// Calculate log-likelihood ratio from game results
    fn log_likelihood_ratio(&self, wins: usize, losses: usize, draws: usize) -> f32 {
        let w = wins as f32;
        let l = losses as f32;
        let d = draws as f32;

        // Expected scores under H0 and H1
        let s0 = Self::elo_to_score(self.elo0);
        let s1 = Self::elo_to_score(self.elo1);

        // Avoid log(0) by using small epsilon
        let eps = 1e-10;
        let s0 = s0.max(eps).min(1.0 - eps);
        let s1 = s1.max(eps).min(1.0 - eps);

        // LLR = sum of log(P(result|H1) / P(result|H0))
        let llr_win = (s1 / s0).ln();
        let llr_loss = ((1.0 - s1) / (1.0 - s0)).ln();
        let llr_draw = 0.5 * (llr_win + llr_loss); // Draw ≈ 0.5 points

        w * llr_win + l * llr_loss + d * llr_draw
    }

    /// Test current game results and return decision
    pub fn test(&self, wins: usize, losses: usize, draws: usize) -> SPRTResult {
        let llr = self.log_likelihood_ratio(wins, losses, draws);
        let games = wins + losses + draws;

        // Decision boundaries
        let lower_bound = (self.beta / (1.0 - self.alpha)).ln();
        let upper_bound = ((1.0 - self.beta) / self.alpha).ln();

        if llr >= upper_bound {
            SPRTResult::H1Accepted { games, llr }
        } else if llr <= lower_bound {
            SPRTResult::H0Accepted { games, llr }
        } else {
            SPRTResult::Continue { games, llr }
        }
    }

    /// Get lower and upper bounds for visualization
    pub fn bounds(&self) -> (f32, f32) {
        let lower = (self.beta / (1.0 - self.alpha)).ln();
        let upper = ((1.0 - self.beta) / self.alpha).ln();
        (lower, upper)
    }
}

pub fn eval_actor_sqrt_from_boards(
    bs: &[Board],
    a1: &impl GetAction,
    a2: &impl GetAction,
    sprt: &SPRT,
    render: bool,
) -> (SPRTResult, f32, f32){
    use crate::board::play_actor_from;
    use indicatif::{ProgressBar, ProgressStyle};

    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;

    let pb = ProgressBar::new((bs.len() * 2) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} \n {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (lower, upper) = sprt.bounds();

    for b in bs {
        // Play one game (a1 starts)
        let (s1, s2) = play_actor_from(b.clone(), a1, a2, render);

        if s1 > s2 {
            wins += 1;
        } else if s2 > s1 {
            losses += 1;
        } else {
            draws += 1;
        }

        // Test after each game
        let result = sprt.test(wins, losses, draws);

        let total_games = wins + losses + draws;
        let score = (wins as f32 + 0.5 * draws as f32) / total_games as f32;

        pb.inc(1);

        match &result {
            SPRTResult::Continue { llr, .. } => {
                pb.set_message(format!(
                    "W/L/D: {}/{}/{} | Score: {:.1}% | LLR: {:.2} [{:.2}, {:.2}]",
                    wins,
                    losses,
                    draws,
                    score * 100.0,
                    llr,
                    lower,
                    upper
                ));
            }
            SPRTResult::H1Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H1 ACCEPTED! A1 is stronger. W/L/D: {}/{}/{} | LLR: {:.2} >= {:.2}",
                    wins, losses, draws, llr, upper
                ));
                return (result, score, 1.0 - score);
            }
            SPRTResult::H0Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H0 ACCEPTED! No significant difference. W/L/D: {}/{}/{} | LLR: {:.2} <= {:.2}",
                    wins, losses, draws, llr, lower
                ));
                return (result, score, 1.0 - score);
            }
            _ => {}
        }

        let (s2, s1) = play_actor_from(b.clone(), a2, a1, render);

        if s1 > s2 {
            wins += 1;
        } else if s2 > s1 {
            losses += 1;
        } else {
            draws += 1;
        }

        // Test after each game
        let result = sprt.test(wins, losses, draws);

        let total_games = wins + losses + draws;
        let score = (wins as f32 + 0.5 * draws as f32) / total_games as f32;

        pb.inc(1);

        match &result {
            SPRTResult::Continue { llr, .. } => {
                pb.set_message(format!(
                    "W/L/D: {}/{}/{} | Score: {:.1}% | LLR: {:.2} [{:.2}, {:.2}]",
                    wins,
                    losses,
                    draws,
                    score * 100.0,
                    llr,
                    lower,
                    upper
                ));
            }
            SPRTResult::H1Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H1 ACCEPTED! A1 is stronger. W/L/D: {}/{}/{} | LLR: {:.2} >= {:.2}",
                    wins, losses, draws, llr, upper
                ));
                return (result, score, 1.0 - score);
            }
            SPRTResult::H0Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H0 ACCEPTED! No significant difference. W/L/D: {}/{}/{} | LLR: {:.2} <= {:.2}",
                    wins, losses, draws, llr, lower
                ));
                return (result, score, 1.0 - score);
            }
            _ => {}
        }
    }

    // Max games reached without conclusion
    let result = SPRTResult::Inconclusive {
        games: wins + losses + draws,
        llr: sprt.log_likelihood_ratio(wins, losses, draws),
    };

    let total_games = wins + losses + draws;
    let score = (wins as f32 + 0.5 * draws as f32) / total_games as f32;

    pb.finish_with_message(format!(
        "⚠ INCONCLUSIVE after {} games. W/L/D: {}/{}/{} | Score: {:.1}%",
        bs.len() * 2,
        wins,
        losses,
        draws,
        score * 100.0
    ));

    (result, score, 1.0 - score)
}

/// Evaluate two agents using SPRT with early termination
/// 
/// Returns as soon as statistical significance is reached or max_games is hit.
pub fn eval_actor_sprt(
    a1: &impl GetAction,
    a2: &impl GetAction,
    max_games: usize,
    sprt: &SPRT,
    render: bool,
) -> (SPRTResult, f32, f32) {
    use crate::board::play_actor;
    use indicatif::{ProgressBar, ProgressStyle};

    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;

    let pb = ProgressBar::new(max_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} \n {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let (lower, upper) = sprt.bounds();

    for i in 0..max_games {
        // Play one game (a1 starts)
        let (s1, s2) = play_actor(a1, a2, render);

        if s1 > s2 {
            wins += 1;
        } else if s2 > s1 {
            losses += 1;
        } else {
            draws += 1;
        }

        // Test after each game
        let result = sprt.test(wins, losses, draws);

        let total_games = wins + losses + draws;
        let score = (wins as f32 + 0.5 * draws as f32) / total_games as f32;

        pb.inc(1);

        match &result {
            SPRTResult::Continue { llr, .. } => {
                pb.set_message(format!(
                    "W/L/D: {}/{}/{} | Score: {:.1}% | LLR: {:.2} [{:.2}, {:.2}]",
                    wins,
                    losses,
                    draws,
                    score * 100.0,
                    llr,
                    lower,
                    upper
                ));
            }
            SPRTResult::H1Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H1 ACCEPTED! A1 is stronger. W/L/D: {}/{}/{} | LLR: {:.2} >= {:.2}",
                    wins, losses, draws, llr, upper
                ));
                return (result, score, 1.0 - score);
            }
            SPRTResult::H0Accepted { llr, .. } => {
                pb.finish_with_message(format!(
                    "✓ H0 ACCEPTED! No significant difference. W/L/D: {}/{}/{} | LLR: {:.2} <= {:.2}",
                    wins, losses, draws, llr, lower
                ));
                return (result, score, 1.0 - score);
            }
            _ => {}
        }
    }

    // Max games reached without conclusion
    let result = SPRTResult::Inconclusive {
        games: wins + losses + draws,
        llr: sprt.log_likelihood_ratio(wins, losses, draws),
    };

    let total_games = wins + losses + draws;
    let score = (wins as f32 + 0.5 * draws as f32) / total_games as f32;

    pb.finish_with_message(format!(
        "⚠ INCONCLUSIVE after {} games. W/L/D: {}/{}/{} | Score: {:.1}%",
        max_games,
        wins,
        losses,
        draws,
        score * 100.0
    ));

    (result, score, 1.0 - score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprt_basic() {
        let sprt = SPRT::new();

        // Test with clear win
        let result = sprt.test(20, 5, 0);
        assert!(matches!(result, SPRTResult::H1Accepted { .. }));

        // Test with clear loss
        let result = sprt.test(5, 20, 0);
        assert!(matches!(result, SPRTResult::H0Accepted { .. }));

        // Test inconclusive
        let result = sprt.test(10, 10, 0);
        assert!(matches!(result, SPRTResult::Continue { .. }));
    }

    #[test]
    fn test_elo_to_score() {
        // Elo +400 should give ~0.91 expected score
        let score = SPRT::elo_to_score(400.0);
        assert!((score - 0.909).abs() < 0.01);

        // Elo 0 should give 0.5
        let score = SPRT::elo_to_score(0.0);
        assert!((score - 0.5).abs() < 0.001);
    }
}
