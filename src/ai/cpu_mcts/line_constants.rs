/// Cell index: cell = z*16 + y*4 + x, x/y/z ∈ {0,1,2,3}
///
/// 76 lines:
///  0-15:  X-lines  (z*4+y,  step=1,  cells z*16+y*4 + {0,1,2,3})
/// 16-31:  Y-lines  (16+z*4+x, step=4,  cells z*16+x + {0,4,8,12})
/// 32-47:  Z-lines  (32+y*4+x, step=16, cells y*4+x + {0,16,32,48})
/// 48-51:  XY-diag1 (48+z,    step=5)
/// 52-55:  XY-diag2 (52+z,    step=3)
/// 56-59:  XZ-diag1 (56+y,    step=17)
/// 60-63:  XZ-diag2 (60+y,    step=15)
/// 64-67:  YZ-diag1 (64+x,    step=20)
/// 68-71:  YZ-diag2 (68+x,    step=12)
/// 72-75:  Space diagonals

const fn get_line_cells(line_id: usize) -> [u8; 4] {
    if line_id < 16 {
        let z = line_id / 4;
        let y = line_id % 4;
        let base = (z * 16 + y * 4) as u8;
        [base, base + 1, base + 2, base + 3]
    } else if line_id < 32 {
        let id = line_id - 16;
        let z = id / 4;
        let x = id % 4;
        let base = (z * 16 + x) as u8;
        [base, base + 4, base + 8, base + 12]
    } else if line_id < 48 {
        let id = line_id - 32;
        let y = id / 4;
        let x = id % 4;
        let base = (y * 4 + x) as u8;
        [base, base + 16, base + 32, base + 48]
    } else if line_id < 52 {
        let z = line_id - 48;
        let base = (z * 16) as u8;
        [base, base + 5, base + 10, base + 15]
    } else if line_id < 56 {
        let z = line_id - 52;
        let base = (z * 16) as u8;
        [base + 3, base + 6, base + 9, base + 12]
    } else if line_id < 60 {
        let y = line_id - 56;
        let base = (y * 4) as u8;
        [base, base + 17, base + 34, base + 51]
    } else if line_id < 64 {
        let y = line_id - 60;
        let base = (y * 4) as u8;
        [base + 3, base + 18, base + 33, base + 48]
    } else if line_id < 68 {
        let x = (line_id - 64) as u8;
        [x, x + 20, x + 40, x + 60]
    } else if line_id < 72 {
        let x = (line_id - 68) as u8;
        [x + 12, x + 24, x + 36, x + 48]
    } else {
        match line_id {
            72 => [0, 21, 42, 63],
            73 => [3, 22, 41, 60],
            74 => [12, 25, 38, 51],
            _  => [15, 26, 37, 48], // 75
        }
    }
}

/// Bitmask of all 4 cells in each line.
pub const LINE_CELL_MASK: [u64; 76] = {
    let mut r = [0u64; 76];
    let mut i = 0;
    while i < 76 {
        let c = get_line_cells(i);
        r[i] = (1u64 << c[0]) | (1u64 << c[1]) | (1u64 << c[2]) | (1u64 << c[3]);
        i += 1;
    }
    r
};

/// For each cell, up to 13 line IDs that cell belongs to; 0xFF = padding.
pub const CELL_TO_LINES: [[u8; 13]; 64] = {
    let mut result = [[0xFFu8; 13]; 64];
    let mut counts = [0u8; 64];
    let mut line_id = 0usize;
    while line_id < 76 {
        let cells = get_line_cells(line_id);
        let mut j = 0;
        while j < 4 {
            let cell = cells[j] as usize;
            let cnt = counts[cell] as usize;
            result[cell][cnt] = line_id as u8;
            counts[cell] += 1;
            j += 1;
        }
        line_id += 1;
    }
    result
};

/// DELTA[cell][word]: add to att_cnt[word] when att places at `cell`.
/// att_cnt packs 76 lines × 2 bits (bit_pos = line_id * 2).
/// 76×2 = 152 bits, fits in 3 u64s with no cross-word issues (2-bit fields
/// never straddle a 64-bit boundary since 2l is always even and max offset = 150).
pub const DELTA: [[u64; 3]; 64] = {
    let mut result = [[0u64; 3]; 64];
    let mut cell = 0usize;
    while cell < 64 {
        let lines = CELL_TO_LINES[cell];
        let mut k = 0;
        while k < 13 {
            let line_id = lines[k];
            if line_id == 0xFF { break; }
            let bit_pos = line_id as usize * 2;
            let word    = bit_pos / 64;
            let offset  = bit_pos % 64;
            result[cell][word] |= 1u64 << offset;
            k += 1;
        }
        cell += 1;
    }
    result
};
