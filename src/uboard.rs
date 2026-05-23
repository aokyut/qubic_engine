pub type UBoard = (u64, u64);
pub type Action = u64;
pub type HalfBoard = u64;

pub trait UBoardActions{
    fn get_sgf(&self) -> (u64, u64, u64);
}

impl UBoardActions for UBoard{
    #[inline(always)]
    fn get_sgf(&self) -> (u64, u64, u64) {
        let stone = self.0 | self.1;
        let ground = !stone & ((stone << 16) | 0xffff);
        let float = !stone & !ground;
        return (stone, ground, float);
    }
}

pub trait HalfBoardActions{
    fn get_lsb(&self) -> Action;
}

impl HalfBoardActions for HalfBoard{
    #[inline(always)]
    fn get_lsb(&self) -> Action {
        return self & (!self + 1);
    }
}