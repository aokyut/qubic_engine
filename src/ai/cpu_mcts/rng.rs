pub struct Xorshift64(u64);

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}
