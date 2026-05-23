# 立体四目並べAI

## 遊び方
* rustをコンパイするためにcargoをインストールする
* qubic_engineディレクトリ内にて"cargo run --release --features=view"で動きます


## 探索関数
### principle_variation_search
* **入力** 
    - const IS_PV: bool...現ノードがPVノードであるか
    - const IS_TRUE_ATT: bool...現ノードが最初に呼び出された攻撃側であるか、最初に呼び出されたときの手番OR実際の白黒
    - (att, def): UBoard: (u64, u64)...現盤面
    - info: &LineInfo...盤面解析のための集積情報
    - hash: u64...現盤面のハッシュ
    - depth: u8...この盤面からの探索深度
    - ply: u8...この盤面までに進んだ深度
    - max_depth: u8...ルートからの葉までの深さ
    - alpha: f32...探索の下界。
    - beta: f32...探索の上界。これを超えたらベータカットが起こる
    - move_history_stack: &mut Vec<u8>...history heuristicに用いる
    - tt: &mut TT...トランスポジションテーブル
    - stats: &mut SearchStats...move orderに用いる統計量。ヒストリーの更新の際は値1 << 14を超えないように加算処理を実装する
        - killer_moves: [[Option<u8>;2];64]...plyごとに管理
        - move_history: [i16; 128]...単純にベストムーブの頻度[0~63]はIS_TRUE_ATT
        - counter_history: [[i16;64];128]...応手の頻度
        - continuous_history2: [[i16;64];128]...自分の前の手に対する頻度
        - continuous_history4: [[i16;64];128]...その前の手に対する頻度
        - continuous_history6: [[i16;64];128]...更に前の手に対する頻度
    - profiler: &mut SearchProfiler...探索の改善用にデータを集めるプロファイラ。null window searchの失敗確率を調べる
    - rng: &mut impl Rng...ランダム源
* **出力**
    - action: u64...アクションマスク
    - fail_val: Fail

* **アルゴリズム**
    1. valid_action_mask <- create_valid_action_mask_from_uboard(att, def)
    1. valid_action_mask <- valid_action_mask ^ blocking_move_mask (即負けの手を排除)
    1. winnig move / blocking move(即勝ちなら出力、即負けなら)
    1. TTEntry <- TT.get(hash)
        1. alpha-beta窓を使って探索を行う
        1. PVノードの時、
            1. 何もしない
        1. NotPVノード And tt_depth >= depthの時
            1. High(x) => 
                beta <= x => return High(x)
            1. Low(x) =>  
                alpha >= x => return Low(x) 
            1. Ex(x) => 
                return Ex(x)
        1. その後にbest_moveを取ってきて行動探索
    1. TTからbestなアクションを取ってきて探索
        1. PVノードのときはカットオフを起こさない（これより深いノードのTTは存在しないので調べる必要もない）
    1. killer moveの探索
        1. killer0が存在するか、その場合にはvalid_mask & killer_mask != 0
            1. 探索
        1. killer1でも同様
        1. killer moveの初期値は64に設定しておく
    1. history moveの探索(このあたりにラインの数による加重を行いたい)
        1. 全てのアクションを列挙してソーティング
        2. それぞれの手について探索

* ** 探索統計量 **
    * cut_offの場所
        * cutノード固有
            * tt_entryによる探索短縮
                * tt_depth >  depth のカウント
                * tt_depth == depth のカウント
                * 下が多ければ>を消すことで探索の厳密性を確保する。
    * ttのbest_moveがどれぐらいの割合でbest_actionだったか知りたい
        * そもそもttにhitする確率
        * ttにhitした中でbest_moveが存在する
        * それがBestである確率
    * tt.insert_forceの前
        * is_pv,depth,tt_hitの有無。
    * killer_moveの的中率を知りたい
        * killer_moveが有効だった数 Hit
        * killer_moveでcutoffが起きた数 CutOff
    * continuous_historyの重みがどれがベストなのか知りたい
    * call-cutは一対一の関係。callに対してCutが最低でも一つ存在する

* **探索部分**
    1. ttあり & depth - 1 >= old_depth
        1. (PVノード以外)アルファ、ベータに対するカットオフ確認
    1. ttなし

* **pv_searchの出力の挙動**
    1. Highを返すとき
        1. if max_value < value and beta < max_valueを起こす => bestな行動が存在する
    1. Exを返す時
        1. 完全探索をして、かつ窓の範囲に収まる => best_move あり
    1. Low
        1. 一度もalphaを超えていない時 => best_moveなしの可能性


## TODO
### line_acumlator.rs
- pv_search_lineinfo
    - LineInfo.nextをlazyに処理
