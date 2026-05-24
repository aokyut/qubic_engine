# 立体四目並べAI

## 遊び方
* rustをコンパイするためにcargoをインストールする
* qubic_engineディレクトリ内にて"cargo run --release --features=view"で動きます


## 探索関数
### principle_variation_search
自己対局におけるNPSは
```
[NPS]1473690030[node]/118084894[μs](12479919[nps])
```
実行環境は
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   16
  On-line CPU(s) list:    0-15
Vendor ID:                GenuineIntel
  Model name:             13th Gen Intel(R) Core(TM) i7-13620H
    CPU family:           6
    Model:                186
    Thread(s) per core:   2
    Core(s) per socket:   8
    Socket(s):            1
    Stepping:             2
    BogoMIPS:             5836.80
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp
                          lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1
                           sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp
                           ibrs_enhanced tpr_shadow ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha
                          _ni xsaveopt xsavec xgetbv1 xsaves avx_vnni vnmi umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize
                           flush_l1d arch_capabilities
Virtualization features:
  Virtualization:         VT-x
  Hypervisor vendor:      Microsoft
  Virtualization type:    full
Caches (sum of all):
  L1d:                    384 KiB (8 instances)
  L1i:                    256 KiB (8 instances)
  L2:                     10 MiB (8 instances)
  L3:                     24 MiB (1 instance)
NUMA:
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-15
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Vulnerable: No microcode
  Retbleed:               Mitigation; Enhanced IBRS
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

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
    - profileでhistoryを調整
    - LineInfo.make .unmakeを実装……&mut selfで自身を書き換えるタイプのnext。探索の度にunmakeが必要だが、それがどの程度処理を圧迫するか読めないが現時点でnextの処理が4.23%、LineInfoのキャッシュミスによる遅れが18.5%あるのでこっちの方式の方が良さそうと思われる。あとは毎度unmakeするのではなく、一手戻して一手進める、ようにすれば処理速度はさらに上がりそう、というかnextとほぼ変わらない速度でできそうな気がする。
    - Actionのソートについて、行動を持ってくるときに最大値を取り出す形の方が良さそう
- mctsの高速化（暇だったら）
    - 差分更新によるロールアウトの激高速化