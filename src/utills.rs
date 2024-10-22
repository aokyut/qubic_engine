pub fn half_imcomplete_beta_func(gamma: f64, delta: f64) -> f64 {
    let mut c0 = 1.0;
    // (g + d + 1)!/(g!d!) * (1/2)^(g + d + 1)

    let mut stack = vec![(gamma, delta)];

    loop {
        // print!("flag");
        let (g, d) = stack.pop().unwrap();

        if g == d && d == 0.0 {
            break;
        }
        if g < d {
            c0 *= 0.5 * (g + d + 1.0) / d;
            stack.push((g, d - 1.0));
        } else {
            c0 *= 0.5 * (g + d + 1.0) / g;
            stack.push((d, g - 1.0));
        }
    }
    c0 *= 0.5;

    let mut a = c0 / (gamma + delta + 1.0);

    for i in 1..=(delta as usize) {
        // println!("{i}");
        a = (a * (i as f64) + c0) / (gamma + delta + 1.0 - i as f64);
    }

    return a;
}
