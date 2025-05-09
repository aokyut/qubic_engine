#[allow(warnings)]
pub mod ai;
pub mod board;
pub mod db;
pub mod exp;
pub mod ml;
pub mod tests;
pub mod train;
pub mod utills;


pub fn add(left: usize, right: usize) -> usize {
    left + right
}
