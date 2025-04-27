// zobrist hashing
/*
・u128->(u128, Fail)
・ハッシュテーブルのサイズは2の冪乗で遷移
・開番地法（線形探索法）とメモ化を使用
*/

use super::Fail;
use rand::{thread_rng, Rng};
use std::alloc::{self, Layout};
use std::fmt;
use std::ptr::NonNull;

pub(crate) const ZHASH_LIST: [usize; 128] = [
    18241299740878438578,
    5374662956693336824,
    13459775845538486322,
    2526521864669311666,
    17515037385517730527,
    614041695805222434,
    3142997059338175354,
    2729993360894845229,
    1640040443000277361,
    5422281586366885909,
    1070943506278138597,
    10072718245807993466,
    14574154717300787730,
    3812251778588158509,
    2842390103177170236,
    8533276189781011437,
    3293818363038847269,
    2538240845391930022,
    13800426143403404609,
    6404717742058983152,
    11230228511605038903,
    1267514535346761347,
    8116007665623266306,
    18308094788061344922,
    5482513084289132630,
    2938388054891600870,
    9163199316621965053,
    13552924035262717395,
    17437794730982494727,
    125292832185831810,
    256600455546787912,
    8988005075020090485,
    7365726000410065520,
    782743999708688822,
    3459310952068207496,
    16396623391978907897,
    16648528756139481589,
    13363452366869139390,
    6292354011865180088,
    15711245569019396874,
    346980645893156641,
    10127064334690385007,
    16634054889382568923,
    3399795960367087676,
    5093598909901385131,
    13823832191482648553,
    14339492030737229414,
    10191122557369820454,
    4303369355729178662,
    5066274577019534792,
    15107362074590740764,
    11963023006955156023,
    17012000629639688499,
    3914171297680965597,
    2921395582608819028,
    15337221847752786016,
    2443210118960994234,
    16689936147173463724,
    2597139145132976108,
    163514754875114172,
    5044899445215194900,
    745748344939276534,
    12275110786485101426,
    16082295474177989800,
    4376834107870886216,
    16281017238810990722,
    1678035397734531907,
    6849185338005199478,
    4119467056907500450,
    11559788215100539369,
    4322444904674809439,
    11486736245458338855,
    18321554705791276418,
    1390747633550857466,
    16936054139007294134,
    8024184199918989128,
    4143021493352759352,
    17103477237798591911,
    16375406066988489289,
    8084008425045304223,
    17773923994682481392,
    7714870911957419566,
    4887679032553321543,
    7756121529419272655,
    2056108801072601364,
    18227467099369805005,
    6235625585579290527,
    8080873997204280530,
    12044939735222900573,
    14186740682732508503,
    7873294212034845830,
    15428961287194532802,
    9899107335290591933,
    17736608632073200363,
    12362350041480994264,
    1335924680467201306,
    8739450330060263396,
    11523477944730503069,
    6702230952947269869,
    2361601486356705466,
    13350451875086594370,
    3682883523428915634,
    4649005115436523599,
    11914519355450447954,
    6508573460573448556,
    11996534714832526785,
    1708933531510522569,
    7140659738240244332,
    7682283615852510581,
    3255397284498339295,
    15829878920569952350,
    18291243138870568561,
    1118520226923253383,
    16191627361390576021,
    13313537335439857266,
    16338862992892466435,
    13874444916871901323,
    123025104241612100,
    10053089266707481511,
    16549836412927111827,
    11100003836048421294,
    6322035389017901649,
    9413371174462444444,
    248823195077864979,
    5059538692691423113,
    18096663533025973035,
    4713459897762151104,
    11797868658862706548,
];

pub(crate) const BUFFER: usize = 100;

pub fn get_hash(board: u128) -> usize {
    let mut hash = 0;
    for i in 0..128 {
        if (board >> i) & 1 == 1 {
            hash ^= ZHASH_LIST[i];
        }
    }
    return hash;
}

pub fn get_diff_hash(diff_board: u128, base_hash: usize) -> usize {
    //! 立体四目用の専用関数
    //! diff_board: 二つだけビットが立ったu128
    //! base_hash: 二手前のボードのハッシュ値
    assert!(diff_board.count_ones() == 2);
    let left = ZHASH_LIST[diff_board.trailing_zeros() as usize];
    let right = ZHASH_LIST[(127 - diff_board.leading_zeros()) as usize];
    return left ^ right ^ base_hash;
}

#[derive(Clone)]
pub struct ZHashEntry {
    memo: usize,
    board: u128,
    val: Fail,
}

const ENTRY_SIZE: usize = 32;
const ENTRY_ALIGNMENT: usize = 16;
const LOAD_FACTOR: usize = 85;

impl ZHashEntry {
    pub(crate) const EMPTY: u8 = 0b11111111;

    pub fn new(memo: usize, board: u128, val: Fail) -> Self {
        return ZHashEntry {
            memo: memo,
            board: board,
            val: val,
        };
    }

    pub fn is_equal(&self, other: ZHashEntry) -> bool {
        return self.memo == other.memo;
    }
}

impl fmt::Debug for ZHashEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = writeln!(
            f,
            "memo:{:x}, \nboard:{:x}, \nval:{:#?}",
            self.memo, self.board, self.val
        )?;

        Ok(())
    }
}

pub fn test_zhash() {
    use rand::thread_rng;
    use rand::Rng;
    let mut hashmap = ZHashMap::new(5);

    hashmap.insert(0, 0xf, Fail::Ex(0.5));
    let a = hashmap.get(2, 0xffff);

    hashmap.all();
}

#[derive(Debug)]
pub struct ZHashMap {
    ptr: NonNull<ZHashEntry>,
    cap: usize,
    pub len: usize,
    ln_size: usize,
    hash_mask: usize,
    //
    pub zhash_list: [usize; 128],
    pub count: usize,
    pub sum_square: usize,
    pub sum: usize,
}

pub fn load_factor_score(count: usize, sum: usize, sum_square: usize) -> f32 {
    let mean = (sum as f32) / (count as f32);
    let var = (sum_square as f32) / (count as f32) - mean * mean;
    let std = var.sqrt();
    return mean + std;
}

impl ZHashMap {
    pub fn new(ln_size: usize) -> Self {
        let cap = (1 << ln_size) + BUFFER;
        let layout = Layout::array::<ZHashEntry>(cap).unwrap();
        let allocated_ptr = unsafe { alloc::alloc(layout) };

        let ptr = match NonNull::new(allocated_ptr as *mut ZHashEntry) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };

        unsafe {
            ptr.write_bytes(ZHashEntry::EMPTY, cap);
        }
        let mut a = [0; 128];
        let mut rng = thread_rng();
        for i in 0..128 {
            let n = rng.gen::<usize>() & ((1 << 63) - 1);
            a[i] = n;
        }

        if cfg!(feature = "render") {
            for i in 0..cap {
                let entry: ZHashEntry = unsafe { ptr.as_ptr().add(i).read() };
                println!("flag:{:#?}, cap:{}, idx:{}", entry, cap, i);
            }

            println!("{}", (1 << ln_size) - 1);
            println!("hash_list:{:#?}", a);
        }

        ZHashMap {
            ptr: ptr,
            cap: cap,
            len: 0,
            ln_size: ln_size,
            hash_mask: (1 << ln_size) - 1,
            //
            zhash_list: a,
            count: 0,
            sum_square: 0,
            sum: 0,
        }
    }

    fn get(&mut self, hash: usize, board: u128) -> Option<Fail> {
        // hashからマスクをとってあーだこーだする
        if cfg!(feature = "render") {
            println!(
                "call get:{hash:x}, board{board:x}, factor:{}",
                self.len * 100 / (1 << self.ln_size)
            );
        }
        let mut idx = hash & self.hash_mask;
        let mut count = 0;
        loop {
            if cfg!(feature = "render") {
                count += 1;
                println!("access {idx}/{}", self.cap);
            }
            let entry: ZHashEntry = unsafe { self.ptr.as_ptr().add(idx).read() };
            if entry.memo == usize::MAX {
                if cfg!(feature = "render") {
                    let factor = self.len * 100 / (1 << self.ln_size);
                    if factor >= 10 && factor < 20 {
                        self.count += 1;
                        self.sum_square += count * count;
                        self.sum += count;
                        let mean = (self.sum as f32) / (self.count as f32);
                        let var = (self.sum_square as f32) / (self.count as f32) - mean * mean;
                        let std = var.sqrt();
                        println!("[lf:{:.2}]{mean:.2}±{std}", self.count);
                    }
                }
                return None;
            } else if entry.board == board {
                if cfg!(feature = "render") {
                    let factor = self.len * 100 / (1 << self.ln_size);
                    if factor >= 10 && factor < 12 {
                        self.count += 1;
                        self.sum_square += count * count;
                        self.sum += count;
                        let mean = (self.sum as f32) / (self.count as f32);
                        let var = (self.sum_square as f32) / (self.count as f32) - mean * mean;
                        let std = var.sqrt();
                        println!("[lf:{count}]{mean}±{std}");
                    }
                }
                return Some(entry.val);
            } else {
                idx += 1;
            }
        }
    }

    fn insert(&mut self, hash: usize, board: u128, val: Fail) {
        // assert!(self.len * 100 / self.cap < 90);
        if cfg!(feature = "render") {
            println!(
                "insert:{hash:x}, fill rate:{}/{}={}",
                self.len,
                1 << self.ln_size,
                self.len * 100 / (1 << self.ln_size)
            );
        }
        if self.len * 100 / (1 << self.ln_size) > LOAD_FACTOR {
            if cfg!(feature = "render") {
                println!("grow:{}/{}", self.len, self.cap);
            }
            self.grow();
        }
        let mut idx = hash & self.hash_mask;

        let new_entry = ZHashEntry {
            memo: hash,
            board: board,
            val: val,
        };
        loop {
            let entry: ZHashEntry = unsafe { self.ptr.as_ptr().add(idx).read() };
            if entry.memo == usize::MAX {
                if cfg!(feature = "render") {
                    println!("[{}] add {idx},{:#?}", self.len, new_entry.clone());
                }
                unsafe { self.ptr.as_ptr().add(idx).write(new_entry) };
                self.len += 1;
                return;
            } else if entry.board == new_entry.board {
                if cfg!(feature = "render") {
                    println!("[{}] add {idx},{:#?}", self.len, new_entry.clone());
                }
                unsafe { self.ptr.as_ptr().add(idx).write(new_entry) };
                return;
            } else {
                idx += 1;
            }
        }
    }

    fn grow(&mut self) {
        if cfg!(feature = "render") {
            println!("grow");
            self.all();
        }

        let cap_half = self.cap - BUFFER;
        let cap = ((self.cap - BUFFER) << 1) + BUFFER;
        let layout = Layout::array::<ZHashEntry>(cap).unwrap();
        let old_layout = Layout::array::<ZHashEntry>(self.cap).unwrap();
        let ptr =
            unsafe { alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, layout.size()) };

        unsafe {
            ptr.add(ENTRY_SIZE * self.cap)
                .write_bytes(ZHashEntry::EMPTY, ENTRY_SIZE * cap_half);
        }

        self.cap = cap;
        self.ln_size += 1;
        self.hash_mask = (1 << self.ln_size) - 1;
        self.ptr = match NonNull::new(ptr as *mut ZHashEntry) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout),
        };
        self.rehash();
        if cfg!(feature = "render") {
            println!("grow after");
            self.all();
        }
    }

    fn rehash(&mut self) {
        for idx in 0..self.cap {
            let entry = unsafe { self.ptr.as_ptr().add(idx).read() };
            if entry.memo == usize::MAX {
                continue;
            }
            let hash = entry.memo & self.hash_mask;
            if idx == hash {
                continue;
            }
            let mut idx_ = hash;
            loop {
                let entry: ZHashEntry = unsafe { self.ptr.as_ptr().add(idx_).read() };
                if cfg!(feature = "render") {
                    println!(
                        "flag:{:#?}, cap:{}, idx:{}, idx_:{}",
                        entry, self.cap, idx, idx_
                    );
                }
                if entry.memo == usize::MAX {
                    unsafe {
                        std::mem::swap(
                            self.ptr.as_ptr().add(idx).as_mut().unwrap(),
                            self.ptr.as_ptr().add(idx_).as_mut().unwrap(),
                        )
                    };
                    if cfg!(feature = "render") {
                        println!("push:{}->{}", idx, idx_);
                    }
                    break;
                } else if idx_ == idx {
                    break;
                } else {
                    idx_ += 1;
                }
            }
        }
    }

    fn all(&self) {
        let mut count = 0;
        for idx in 0..self.cap {
            let entry = unsafe { self.ptr.as_ptr().add(idx).read() };
            if cfg!(feature = "render") {
                println!("flag:{:#?}, cap:{}, idx:{}", entry, self.cap, idx);
            }
            if entry.memo != usize::MAX {
                count += 1;
            }
        }
        assert_eq!(self.len, count);
    }
}

impl Drop for ZHashMap {
    fn drop(&mut self) {
        if self.cap != 0 {
            let layout = Layout::array::<ZHashEntry>(self.cap).unwrap();
            unsafe {
                // heap::deallocate(*self.ptr as *mut _, num_bytes, align);
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

unsafe impl Sync for ZHashMap {}
unsafe impl Send for ZHashMap {}

use super::EvaluatorF;
use crate::board::Board;
pub fn negalphaf_zhash(
    parent_hash_pare: Option<(u128, usize)>,
    b: &Board,
    b_hash_pare: (u128, usize),
    depth: u8,
    alpha: f32,
    beta: f32,
    hashmap: &mut ZHashMap,
    e: &Box<dyn EvaluatorF>,
) -> (u8, Fail, i32) {
    use crate::ai::b2u128;
    use Fail::*;

    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -2.0;
    let mut max_actions = Vec::new();
    let mut alpha = alpha;

    // pprint_board(b);
    // print_blank(5 - depth);
    // println!("[depth:{depth}]alpha:{alpha}, beta:{beta}");

    if depth <= 1 {
        for action in actions.iter() {
            let next_board = &b.next(*action);
            let next_bitboard = b2u128(&next_board);
            // let hash = next_board.hash();
            let next_hash = match parent_hash_pare {
                Some((bitboard, hash)) => {
                    // println!("bboard:{bitboard:x}, next_bboard:{next_bitboard:x}, hash:{hash:x}, next_hash:{:x}", get_diff_hash(next_bitboard ^ bitboard, hash));
                    get_diff_hash(next_bitboard ^ bitboard, hash)
                }
                None => get_hash(next_bitboard),
            };
            let map_val = hashmap.get(next_hash, next_bitboard);
            let val;
            match map_val {
                Some(old_val) => {
                    if old_val.is_equal(0.0) {
                        return (*action, Ex(1.0), count);
                    }
                    val = 1.0 - old_val.get_val();
                }
                None => {
                    if next_board.is_win() {
                        hashmap.insert(next_hash, next_bitboard, Ex(0.0));
                        return (*action, Ex(1.0), count);
                    } else if next_board.is_draw() {
                        hashmap.insert(next_hash, next_bitboard, Ex(0.5));
                        return (*action, Ex(0.5), count);
                    }
                    let next_val = e.eval_func_f32(next_board);
                    val = 1.0 - next_val;
                    hashmap.insert(next_hash, next_bitboard, Ex(next_val));
                }
            }
            if max_val < val {
                max_val = val;
                max_actions = vec![*action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (*action, High(max_val), count);
                    }
                }
            } else if max_val == val {
                max_actions.push(*action);
            }
        }
    } else {
        let mut action_nb_vals: Vec<(u8, Board, f32, (u128, usize), Option<Fail>)> = Vec::new();

        for action in actions.into_iter() {
            let next_board = b.next(action);
            // let hash = next_board.hash();
            let next_bitboard = b2u128(&next_board);
            let next_hash = match parent_hash_pare {
                Some((bitboard, hash)) => get_diff_hash(bitboard ^ next_bitboard, hash),
                None => get_hash(next_bitboard),
            };
            let map_val = hashmap.get(next_hash, next_bitboard);

            match map_val {
                Some(old_val) => {
                    if old_val.is_equal(0.0) {
                        return (action, Ex(1.0), count);
                    }
                    action_nb_vals.push((
                        action,
                        next_board,
                        old_val.f32_minus(1.0).get_val(),
                        (next_bitboard, next_hash),
                        Some(old_val.inverse()),
                    ));
                }
                None => {
                    if next_board.is_win() {
                        hashmap.insert(next_hash, next_bitboard, Ex(0.0));
                        // print_blank(5 - depth);
                        // println!("[depth:{depth}]action:{action}, win");
                        return (action, Ex(1.0), count);
                    } else if next_board.is_draw() {
                        hashmap.insert(next_hash, next_bitboard, Ex(0.5));
                        // print_blank(5 - depth);
                        // println!("[depth:{depth}]action:{action}, draw");
                        return (action, Ex(0.5), count);
                    }
                    let val = 1.0 - e.eval_func_f32(&next_board);
                    action_nb_vals.push((
                        action,
                        next_board,
                        val,
                        (next_bitboard, next_hash),
                        None,
                    ));
                }
            }
        }

        action_nb_vals.sort_by(|a, b| {
            // a.2.cmp(&b.2).reverse();
            b.2.partial_cmp(&a.2).unwrap()
        });

        for (action, next_board, old_val, (n_bboard, n_hash), hit) in action_nb_vals {
            let val;
            // println!("[depth:{depth}], action:{action}, alpha:{alpha}, beta:{beta}",);
            if let Some(fail_val) = hit {
                // if depth > 2 {
                //     println!("[{depth}]hit, {}", fail_val.to_string());
                // }
                match fail_val {
                    High(x) => {
                        if beta < x {
                            return (action, High(x), count);
                        } else {
                            let new_alpha = x.max(alpha);
                            let (_, _val, _count) = negalphaf_zhash(
                                Some(b_hash_pare),
                                &next_board,
                                (n_bboard, n_hash),
                                depth - 1,
                                1.0 - beta,
                                1.0 - new_alpha,
                                hashmap,
                                e,
                            );
                            count += _count;
                            hashmap.insert(n_hash, n_bboard, _val);
                            let _val = _val.inverse();
                            if _val.is_fail_high() {
                                return (action, High(beta), count);
                            }
                            val = _val.get_val();
                        }
                    }
                    Low(x) => {
                        if alpha > x {
                            continue;
                        } else {
                            let new_beta = x.min(beta);
                            // fial_low(alpha) or fail_ex(val) < new_beta
                            let (_, _val, _count) = negalphaf_zhash(
                                Some(b_hash_pare),
                                &next_board,
                                (n_bboard, n_hash),
                                depth - 1,
                                1.0 - new_beta,
                                1.0 - alpha,
                                hashmap,
                                e,
                            );
                            hashmap.insert(n_hash, n_bboard, _val);
                            let _val = _val.inverse();
                            if _val.is_fail_low() {
                                continue;
                            }
                            val = _val.get_val();
                        }
                    }
                    Ex(x) => {
                        // val = F32_INVERSE_BIAS - F32_INVERSE_BIAS * x;
                        val = x;
                    }
                }
            } else {
                let (_, _val, _count) = negalphaf_zhash(
                    Some(b_hash_pare),
                    &next_board,
                    (n_bboard, n_hash),
                    depth - 1,
                    1.0 - beta,
                    1.0 - alpha,
                    hashmap,
                    e,
                );
                hashmap.insert(n_hash, n_bboard, _val);
                count += 1 + _count;
                let _val = _val.inverse();

                match _val {
                    High(x) => return (action, High(x), count),
                    Low(_) => continue,
                    Ex(x) => {
                        val = x;
                    }
                }
            }
            if max_val < val {
                max_val = val;
                max_actions = vec![action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (action, High(max_val), count);
                    }
                }
            } else if max_val == val {
                max_actions.push(action);
            }
        }
    }
    let mut rng = rand::thread_rng();
    if max_actions.len() == 0 {
        return (201, Low(alpha), count);
    }

    return (
        max_actions[rng.gen::<usize>() % max_actions.len()],
        Ex(max_val),
        count,
    );
}
