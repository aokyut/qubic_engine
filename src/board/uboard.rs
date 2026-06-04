pub type UBoard = (u64, u64);
pub type Action = u64;
pub type HalfBoard = u64;

pub trait UBoardActions{
    fn get_sgf(&self) -> (u64, u64, u64);
    fn is_win(&self) -> bool;
    fn is_draw(&self) -> bool;
    fn get_valid(&self) -> u64; // 可能な行動のマスクを得る。
    fn next(&self, action:u64) -> Self;
}

impl UBoardActions for UBoard{
    #[inline(always)]
    fn get_sgf(&self) -> (u64, u64, u64) {
        let stone = self.0 | self.1;
        let ground = !stone & ((stone << 16) | 0xffff);
        let float = !stone & !ground;
        return (stone, ground, float);
    }
    fn is_win(&self) -> bool{
        let bit = self.0;
        (bit & (bit >> 1) & (bit >> 2) & (bit >> 3) & 0x1111111111111111)
            | (bit & (bit >> 4) & (bit >> 8) & (bit >> 12) & 0x000f000f000f000f)
            | (bit & (bit >> 16) & (bit >> 32) & (bit >> 48) & 0x000000000000ffff)
            | (bit & (bit >> 5) & (bit >> 10) & (bit >> 15) & 0x0001000100010001)
            | (bit & (bit >> 3) & (bit >> 6) & (bit >> 9) & 0x0008000800080008)
            | (bit & (bit >> 17) & (bit >> 34) & (bit >> 51) & 0x1111)
            | (bit & (bit >> 15) & (bit >> 30) & (bit >> 45) & 0x8888)
            | (bit & (bit >> 20) & (bit >> 40) & (bit >> 60) & 0x000f)
            | (bit & (bit >> 12) & (bit >> 24) & (bit >> 36) & 0xf000)
            | (bit & (bit >> 21) & (bit >> 42) & (bit >> 63))
            | (bit & (bit >> 19) & (bit >> 38) & (bit >> 57) & 0x0008)
            | (bit & (bit >> 13) & (bit >> 26) & (bit >> 39) & 0x1000)
            | (bit & (bit >> 11) & (bit >> 22) & (bit >> 33) & 0x8000)
            > 0
    }
    fn is_draw(&self) -> bool {
        ((self.0 | self.1) >> 48) == 0xffff
    }
    fn get_valid(&self) -> u64 {
        let stone = self.0 | self.1;
        let ground = !stone & ((stone << 16) | 0xffff);
        return ground;
    }
    fn next(&self, action:u64) -> Self {
        return (self.1, self.0 | action);
    }
}

const fn _get_reach_mask(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64) -> u64 {
    return d & (b & c & (a | e) | e & f & (c | g));
}

// リーチとなる行動を調べる。
pub fn get_reach_mask(a: u64, d: u64) -> u64 {
    let stone = a | d;
    let blank = !(stone) & ((stone << 16) | 0xffff);
    let x = _get_reach_mask(
        (a >> 3) & 0x1111_1111_1111_1111,
        (a >> 2) & 0x3333_3333_3333_3333,
        (a >> 1) & 0x7777_7777_7777_7777,
        blank,
        (a << 1) & 0xeeee_eeee_eeee_eeee,
        (a << 2) & 0xcccc_cccc_cccc_cccc,
        (a << 3) & 0x8888_8888_8888_8888,
    );
    let y = _get_reach_mask(
        (a >> 12) & 0x000f_000f_000f_000f,
        (a >> 8) & 0x00ff_00ff_00ff_00ff,
        (a >> 4) & 0x0fff_0fff_0fff_0fff,
        blank,
        (a << 4) & 0xfff0_fff0_fff0_fff0,
        (a << 8) & 0xff00_ff00_ff00_ff00,
        (a << 12) & 0xf000_f000_f000_f000,
    );
    let z = _get_reach_mask(a >> 48, a >> 32, a >> 16, blank, a << 16, a << 32, a << 48);
    let xy = _get_reach_mask(
        (a >> 15) & 0x0001_0001_0001_0001,
        (a >> 10) & 0x0033_0033_0033_0033,
        (a >> 5) & 0x0777_0777_0777_0777,
        blank,
        (a << 5) & 0xeee0_eee0_eee0_eee0,
        (a << 10) & 0xcc00_cc00_cc00_cc00,
        (a << 15) & 0x8000_8000_8000_8000,
    );
    let yx = _get_reach_mask(
        (a >> 9) & 0x0008_0008_0008_0008,
        (a >> 6) & 0x00cc_00cc_00cc_00cc,
        (a >> 3) & 0x0eee_0eee_0eee_0eee,
        blank,
        (a << 3) & 0x7770_7770_7770_7770,
        (a << 6) & 0x3300_3300_3300_3300,
        (a << 9) & 0x1000_1000_1000_1000,
    );
    let xz = _get_reach_mask(
        (a >> 51) & 0x0000_0000_0000_1111,
        (a >> 34) & 0x0000_0000_3333_3333,
        (a >> 17) & 0x0000_7777_7777_7777,
        blank,
        (a << 17) & 0xeeee_eeee_eeee_0000,
        (a << 34) & 0xcccc_cccc_0000_0000,
        (a << 51) & 0x8888_0000_0000_0000,
    );
    let zx = _get_reach_mask(
        (a >> 45) & 0x0000_0000_0000_8888,
        (a >> 30) & 0x0000_0000_cccc_cccc,
        (a >> 15) & 0x0000_eeee_eeee_eeee,
        blank,
        (a << 15) & 0x7777_7777_7777_0000,
        (a << 30) & 0x3333_3333_0000_0000,
        (a << 45) & 0x1111_0000_0000_0000,
    );
    let yz = _get_reach_mask(
        (a >> 60) & 0xf,
        (a >> 40) & 0x00ff_00ff,
        (a >> 20) & 0x0fff_0fff_0fff,
        blank,
        (a << 20) & 0xfff0_fff0_fff0_0000,
        (a << 40) & 0xff00_ff00_0000_0000,
        (a << 60) & 0xf000_0000_0000_0000,
    );
    let zy = _get_reach_mask(
        (a >> 36) & 0xf000,
        (a >> 24) & 0xff00_ff00,
        (a >> 12) & 0xfff0_fff0_fff0,
        blank,
        (a << 12) & 0x0fff_0fff_0fff_0000,
        (a << 24) & 0x00ff_00ff_0000_0000,
        (a << 36) & 0x000f_0000_0000_0000,
    );
    let xyz = _get_reach_mask(
        (a >> 63) & 0x1,
        (a >> 42) & 0x0033_0033,
        (a >> 21) & 0x0777_0777_0777,
        blank,
        (a << 21) & 0xeee0_eee0_eee0_0000,
        (a << 42) & 0xcc00_cc00_0000_0000,
        (a << 63) & 0x8000_0000_0000_0000,
    );
    let yzx = _get_reach_mask(
        (a >> 57) & 0x8,
        (a >> 38) & 0x00cc_00cc,
        (a >> 19) & 0x0eee_0eee_0eee,
        blank,
        (a << 19) & 0x7770_7770_7770_0000,
        (a << 38) & 0x3300_3300_0000_0000,
        (a << 57) & 0x1000_0000_0000_0000,
    );
    let xzy = _get_reach_mask(
        (a >> 39) & 0x1000,
        (a >> 26) & 0x3300_3300,
        (a >> 13) & 0x7770_7770_7770,
        blank,
        (a << 13) & 0x0eee_0eee_0eee_0000,
        (a << 26) & 0x00cc_00cc_0000_0000,
        (a << 39) & 0x0008_0000_0000_0000,
    );
    let zyx = _get_reach_mask(
        (a >> 33) & 0x8000,
        (a >> 22) & 0xcc00_cc00,
        (a >> 11) & 0xeee0_eee0_eee0,
        blank,
        (a << 11) & 0x0777_0777_0777_0000,
        (a << 22) & 0x0033_0033_0000_0000,
        (a << 33) & 0x0001_0000_0000_0000,
    );

    return x | y | z | xy | yx | xz | zx | yz | zy | xyz | xzy | yzx | zyx;
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