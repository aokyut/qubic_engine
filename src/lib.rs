#[allow(warnings)]
pub mod ai;
pub mod board;
#[cfg(not(target_arch = "wasm32"))]
pub mod db;

pub mod dfpn;
pub mod exp;
pub mod ml;
pub mod tests;
#[cfg(not(target_arch = "wasm32"))]
pub mod train;
pub mod utills;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}
