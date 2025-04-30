// use super::super::board::GetAction;
// use super::*;

// pub async fn negalpha_wraper(
//     b: &Board,
//     depth: u8,
//     alpha: f32,
//     beta: f32,
//     gen: u8,
//     hashmap: &mut HashMap<u128, (Fail, u8)>,
//     e: Arc<dyn SyncEvaluatorF>,
// ) -> (u8, Fail, i32) {
//     return negalphaf_hash_iter_arc(b, depth, alpha, beta, gen, hashmap, e);
// }

// pub fn negalphaf_hash_iter_arc(
//     b: &Board,
//     depth: u8,
//     alpha: f32,
//     beta: f32,
//     gen: u8,
//     hashmap: &mut HashMap<u128, (Fail, u8)>,
//     e: Arc<dyn SyncEvaluatorF>,
// ) -> (u8, Fail, i32) {
//     use Fail::*;

//     let mut count = 0;
//     let actions = b.valid_actions();
//     let mut max_val = -2.0;
//     let mut max_actions = Vec::new();
//     let mut alpha = alpha;

//     // pprint_board(b);
//     // print_blank(5 - depth);
//     // println!("[depth:{depth}]alpha:{alpha}, beta:{beta}");

//     if depth <= 1 {
//         for action in actions.iter() {
//             let next_board = &b.next(*action);
//             // let hash = next_board.hash();
//             let hash = b2u128(next_board);
//             let map_val = hashmap.get(&hash);
//             let val;
//             match map_val {
//                 Some(&(old_val, old_gen)) => {
//                     if old_val.is_equal(0.0) {
//                         return (*action, Ex(1.0), count);
//                     } else if next_board.is_draw() {
//                         return (*action, Ex(0.5), count);
//                     }
//                     val = 1.0 - old_val.get_val();
//                 }
//                 None => {
//                     if next_board.is_win() {
//                         hashmap.insert(hash, (Ex(0.0), gen));
//                         return (*action, Ex(1.0), count);
//                     } else if next_board.is_draw() {
//                         hashmap.insert(hash, (Ex(0.5), gen));
//                         return (*action, Ex(0.5), count);
//                     }
//                     let next_val = e.eval_func_f32(next_board);
//                     val = 1.0 - next_val;
//                     hashmap.insert(hash, (Ex(next_val), gen));
//                 }
//             }
//             if max_val < val {
//                 max_val = val;
//                 max_actions = vec![*action];
//                 if max_val > alpha {
//                     alpha = max_val;
//                     if alpha > beta {
//                         // println!("[{depth}]->max_val:{max_val}");
//                         return (*action, High(max_val), count);
//                     }
//                 }
//             } else if max_val == val {
//                 max_actions.push(*action);
//             }
//         }
//     } else {
//         let mut action_nb_vals: Vec<(u8, Board, f32, u128, (Option<Fail>, u8))> = Vec::new();

//         for action in actions.into_iter() {
//             let next_board = b.next(action);
//             // let hash = next_board.hash();
//             let hash = b2u128(&next_board);
//             let map_val = hashmap.get(&hash);

//             match map_val {
//                 Some(&(old_val, old_gen)) => {
//                     if old_val.is_equal(0.0) {
//                         return (action, Ex(1.0), count);
//                     } else if next_board.is_draw() {
//                         return (action, Ex(0.5), count);
//                     }
//                     action_nb_vals.push((
//                         action,
//                         next_board,
//                         old_val.f32_minus(1.0).get_val(),
//                         hash,
//                         (Some(old_val.inverse()), old_gen),
//                     ));
//                 }
//                 None => {
//                     if next_board.is_win() {
//                         hashmap.insert(hash, (Ex(0.0), gen));
//                         // print_blank(5 - depth);
//                         // println!("[depth:{depth}]action:{action}, win");
//                         return (action, Ex(1.0), count);
//                     } else if next_board.is_draw() {
//                         hashmap.insert(hash, (Ex(0.5), gen));
//                         // print_blank(5 - depth);
//                         // println!("[depth:{depth}]action:{action}, draw");
//                         return (action, Ex(0.5), count);
//                     }
//                     let val = 1.0 - e.eval_func_f32(&next_board);
//                     action_nb_vals.push((action, next_board, val, hash, (None, 0)));
//                 }
//             }
//         }

//         action_nb_vals.sort_by(|a, b| {
//             // a.2.cmp(&b.2).reverse();
//             b.2.partial_cmp(&a.2).unwrap()
//         });

//         for (action, next_board, old_val, hash, (hit, old_gen)) in action_nb_vals {
//             let val;
//             if old_gen == gen {
//                 if let Some(fail_val) = hit {
//                     // if depth > 2 {
//                     //     println!("[{depth}]hit, {}", fail_val.to_string());
//                     // }
//                     match fail_val {
//                         High(x) => {
//                             if beta < x {
//                                 return (action, High(x), count);
//                             } else {
//                                 let new_alpha = x.max(alpha);
//                                 let (_, _val, _count) = negalphaf_hash_iter_arc(
//                                     &next_board,
//                                     depth - 1,
//                                     1.0 - beta,
//                                     1.0 - new_alpha,
//                                     gen,
//                                     hashmap,
//                                     e.clone(),
//                                 );
//                                 count += _count;
//                                 hashmap.insert(hash, (_val, gen));
//                                 let _val = _val.inverse();
//                                 if _val.is_fail_high() {
//                                     return (action, High(beta), count);
//                                 }
//                                 val = _val.get_val();
//                             }
//                         }
//                         Low(x) => {
//                             if alpha > x {
//                                 continue;
//                             } else {
//                                 let new_beta = x.min(beta);
//                                 // fial_low(alpha) or fail_ex(val) < new_beta
//                                 let (_, _val, _count) = negalphaf_hash_iter_arc(
//                                     &next_board,
//                                     depth - 1,
//                                     1.0 - new_beta,
//                                     1.0 - alpha,
//                                     gen,
//                                     hashmap,
//                                     e.clone(),
//                                 );
//                                 hashmap.insert(hash, (_val, gen));
//                                 let _val = _val.inverse();
//                                 if _val.is_fail_low() {
//                                     continue;
//                                 }
//                                 val = _val.get_val();
//                             }
//                         }
//                         Ex(x) => {
//                             // val = F32_INVERSE_BIAS - F32_INVERSE_BIAS * x;
//                             val = x;
//                         }
//                     }
//                 } else {
//                     let (_, _val, _count) = negalphaf_hash_iter_arc(
//                         &next_board,
//                         depth - 1,
//                         1.0 - beta,
//                         1.0 - alpha,
//                         gen,
//                         hashmap,
//                         e.clone(),
//                     );
//                     hashmap.insert(hash, (_val, gen));
//                     count += 1 + _count;
//                     let _val = _val.inverse();

//                     match _val {
//                         High(x) => return (action, High(x), count),
//                         Low(_) => continue,
//                         Ex(x) => {
//                             val = x;
//                         }
//                     }
//                 }
//             } else {
//                 let (_, _val, _count) = negalphaf_hash_iter_arc(
//                     &next_board,
//                     depth - 1,
//                     1.0 - beta,
//                     1.0 - alpha,
//                     gen,
//                     hashmap,
//                     e.clone(),
//                 );
//                 hashmap.insert(hash, (_val, gen));
//                 count += 1 + _count;
//                 let _val = _val.inverse();

//                 match _val {
//                     High(x) => return (action, High(x), count),
//                     Low(_) => continue,
//                     Ex(x) => {
//                         val = x;
//                     }
//                 }
//             }

//             if max_val < val {
//                 max_val = val;
//                 max_actions = vec![action];
//                 if max_val > alpha {
//                     alpha = max_val;
//                     if alpha > beta {
//                         return (action, High(max_val), count);
//                     }
//                 }
//             } else if max_val == val {
//                 max_actions.push(action);
//             }
//         }
//     }
//     let mut rng = rand::thread_rng();
//     if max_actions.len() == 0 {
//         return (201, Low(alpha), count);
//     }

//     return (
//         max_actions[rng.gen::<usize>() % max_actions.len()],
//         Ex(max_val),
//         count,
//     );
// }

// pub struct TimeoutNegAlphaF {
//     evaluator: Arc<dyn SyncEvaluatorF>,
//     pub timeout: u128,
//     min_depth: u8,
//     max_depth: u8,
//     pub checking_mate: bool,
// }

// impl TimeoutNegAlphaF {
//     fn new(evaluator: Arc<dyn SyncEvaluatorF>, max_depth: u8, min_depth: u8) -> Self {
//         // let evaluator: Arc<dyn EvaluatorF> = Arc::new(evaluator);
//         return TimeoutNegAlphaF {
//             evaluator: evaluator,
//             timeout: 1000,
//             min_depth: min_depth,
//             max_depth: max_depth,
//             checking_mate: false,
//         };
//     }

//     fn eval_with_negalpha<'a>(&'a self, b: &Board) -> (u8, f32, i32) {
//         use std::sync::mpsc;
//         use std::thread;

//         let timer = Instant::now();
//         let b_ = b.clone();

//         let (sender, reciever) = mpsc::channel();
//         let e = Arc::clone(&self.evaluator);

//         let mut action = board::get_random(&b);
//         let mut val = 0.0;
//         let mut count = 0;

//         loop {
//             if let Result::Ok((a, v, c)) = reciever.try_recv() {
//                 action = a;
//                 val = v.get_exval().unwrap();
//                 count = c;
//             }
//             if timer.elapsed().as_millis() >= self.timeout {
//                 return (action, val, count);
//             }
//         }
//     }
// }

// trait SyncEvaluatorF: EvaluatorF + Sync + Send {}

// impl EvaluatorF for TimeoutNegAlphaF {
//     fn eval_func_f32(&self, b: &Board) -> f32 {
//         let (a, b, c) = self.eval_with_negalpha(b);
//         return b;
//     }
// }

// impl GetAction for TimeoutNegAlphaF {
//     fn get_action(&self, b: &Board) -> u8 {
//         let (a, b, c) = self.eval_with_negalpha(b);
//         return a;
//     }
// }
