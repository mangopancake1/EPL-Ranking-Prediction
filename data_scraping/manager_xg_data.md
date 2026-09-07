# Manager xG Data — EPL 2026/27
**Tanggal:** 2026-08-03  
**Format:** xG per 90 menit | MP = Matches Played

---

## METODOLOGI

Setiap manajer baru → ambil data xG dari **klub yang dia latih sebelumnya**.  
League weight diterapkan saat blending ke model Dixon-Coles.  
Data diurutkan dari musim terlama ke terbaru per manajer.

---

## ✅ DATA TERSEDIA

---

### 1. Andoni Iraola → Liverpool
**Prior klub:** AFC Bournemouth (EPL) | **League weight:** 1.0000

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 23/24 | EPL | 38 | 1.37 | 1.40 | -0.03 | 1.42 | 1.76 | +0.05 |
| 24/25 | EPL | 38 | 1.57 | 1.35 | +0.22 | 1.53 | 1.21 | -0.04 |
| 25/26 | EPL | 38 | 1.56 | 1.47 | +0.09 | 1.53 | 1.42 | -0.03 |

**Catatan:** ⚠️ **KOREKSI 2026-08:** baris 22/23 (xG 1.05/1.82) yang sebelumnya ada di sini **dihapus** — itu bukan musim Iraola. Iraola baru masuk Bournemouth Juni 2023, jadi musim pertamanya adalah 23/24. Baris 22/23 itu sebenarnya musim Gary O'Neil (lihat #12 di bawah) — ketauan karena user kasih data yang sama persis buat entri O'Neil terpisah. Tren positif tetap jelas dari 3 musim asli Iraola: xGD +(-0.03) → +0.22 → +0.09. Data lengkap ✓ (dikoreksi)

---

### 2. Xabi Alonso → Chelsea
**Prior klub:** Bayer Leverkusen (Bundesliga) | **League weight:** 0.7866

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | Bundesliga | 34 | 1.42 | 1.32 | +0.10 | 1.68 | 1.44 | +0.26 |
| 23/24 | Bundesliga | 34 | 1.88 | 0.89 | +0.99 | 2.62 | 0.71 | +0.74 |
| 24/25 | Bundesliga | 34 | 1.66 | 1.07 | +0.59 | 2.12 | 1.26 | +0.46 |

**Catatan:** 23/24 adalah musim invincible Leverkusen — outlier ke atas. Model harus hati-hati tidak terlalu terpengaruh satu musim luar biasa ini. Data lengkap ✓

---

### 3. Marco Rose → Bournemouth
**Prior klub:** RB Leipzig (Bundesliga) | **League weight:** 0.7866

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | Bundesliga | 34 | 1.62 | 1.04 | +0.58 | 1.88 | 1.21 | +0.26 |
| 23/24 | Bundesliga | 34 | 1.56 | 1.13 | +0.43 | 2.26 | 1.15 | +0.70 |
| 24/25 | Bundesliga | 34 | 1.27 | 1.40 | -0.13 | 1.56 | 1.41 | +0.29 |

**Catatan:** Tren memburuk — xGD turun dari +0.58 → -0.13. Leipzig 24/25 jauh di bawah performa sebelumnya. Perlu dicek apakah Rose dipecat atau mundur. Data lengkap ✓

---

### 4. Pierre Sage → Crystal Palace
**Prior klub:** Olympique Lyonnais (Ligue 1) | **League weight:** 0.6642  
**Bonus:** Marseille data tersedia (konteks De Zerbi, bukan Sage)

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | Ligue 1 | 38 | 1.64 | 1.29 | +0.35 | 1.71 | 1.24 | +0.07 |
| 23/24 | Ligue 1 | 34 | 1.36 | 1.33 | +0.03 | 1.44 | 1.62 | +0.08 |
| 24/25 | Ligue 1 | 34 | 1.46 | 1.43 | +0.03 | 1.91 | 1.35 | +0.45 |

**Catatan:** 22/23 bukan musim Sage — dia baru masuk Desember 2023 (mid-season 23/24). Data 22/23 adalah era sebelum Sage → **exclude dari prior Sage**. Gunakan 23/24 (partial) dan 24/25 saja. Data partial ⚠

---

### 5. Roberto De Zerbi → Spurs
**Prior klub A:** Brighton (EPL) | **League weight:** 1.0000  
**Prior klub B:** Marseille (Ligue 1) | **League weight:** 0.6642

#### Brighton (De Zerbi):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | EPL | 38 | 1.80 | 1.14 | +0.66 | 1.89 | 1.39 | +0.09 |
| 23/24 | EPL | 38 | 1.46 | 1.20 | +0.26 | 1.45 | 1.63 | -0.01 |
| 24/25 | EPL | 38 | 1.48 | 1.14 | +0.34 | 1.74 | 1.55 | +0.26 |
| 25/26 | EPL | 38 | 1.54 | 1.33 | +0.21 | 1.37 | 1.21 | -0.17 |

**Catatan:** De Zerbi di Brighton mulai 2022/23. Data lengkap 4 musim. ✓

#### Marseille (De Zerbi):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 24/25 | Ligue 1 | 34 | 1.59 | 1.06 | +0.53 | 2.18 | 1.38 | +0.59 |
| 25/26 | Ligue 1 | 34 | 1.72 | 1.18 | +0.54 | 1.85 | 1.32 | +0.13 |

**Catatan:** Data Marseille lengkap untuk 2 musim. Blend: 70% Brighton / 30% Marseille per kesepakatan sebelumnya. ✓

---

### 6. Matthias Jaissle → Newcastle
**Prior klub:** Al-Ahli (Saudi Pro League) | **Weight:** squad value proxy  
~~RB Salzburg — di-skip, data berbayar~~

#### Al-Ahli (Jaissle):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 23/24 | Saudi PL | 34 | 1.55 | 0.89 | +0.66 | 1.97 | 1.03 | +0.42 |
| 24/25 | Saudi PL | 34 | 1.71 | 1.11 | +0.60 | 2.03 | 1.06 | +0.32 |
| 25/26 | Saudi PL | 34 | 1.58 | 0.99 | +0.59 | 2.09 | 0.74 | +0.51 |

**Catatan:** Data Al-Ahli konsisten — xGD stabil di +0.59–0.66 selama 3 musim. Salzburg di-skip (data berbayar). Prior Jaissle = Al-Ahli only, weight via squad value proxy. ✓

---

### 7. Enzo Maresca → Man City
**Prior klub:** Chelsea 25/26 (EPL) | **League weight:** 1.0000

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 24/25 | EPL | 38 | 1.64 | 1.13 | +0.51 | 1.68 | 1.13 | +0.04 |
| 25/26 | EPL | 38 | 1.54 | 1.31 | +0.23 | 1.53 | 1.37 | -0.01 |

**Catatan:** Maresca mulai Chelsea dari 24/25. Data lengkap 2 musim. ✓

---

### 8. Oliver Glasner → Nottingham Forest
**Prior klub:** Crystal Palace (EPL) | **League weight:** 1.0000

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 23/24 | EPL | 38 | 1.19 | 1.25 | -0.06 | 1.50 | 1.53 | +0.31 |
| 24/25 | EPL | 38 | 1.35 | 1.36 | -0.01 | 1.34 | 1.34 | -0.01 |
| 25/26 | EPL | 38 | 1.31 | 1.40 | -0.09 | 1.08 | 1.34 | -0.23 |

**Catatan:** Glasner di Crystal Palace mulai 2023/24. Data lengkap 3 musim. ✓

---

### 9. Michael Carrick → Man United (full season)
**Prior:** Man United sendiri, masuk mid-season 25/26 | **League weight:** 1.0000

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 25/26 | EPL | 38 | 1.74 | 1.36 | +0.38 | 1.82 | 1.32 | +0.08 |

**Catatan:** Data 25/26 mencakup full musim termasuk sebelum Carrick masuk. Perlu split berdasarkan tanggal masuk Carrick untuk isolasi dampaknya. Data partial ⚠

---

### 10. Manajer Lama — Data Relevan Penuh

#### Arsenal (Mikel Arteta):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 23/24 | 38 | 1.69 | 0.79 | +0.90 | 2.39 | 0.76 | +0.70 |
| 24/25 | 38 | 1.61 | 0.98 | +0.63 | 1.82 | 0.89 | +0.21 |
| 25/26 | 38 | 1.70 | 0.95 | +0.75 | 1.87 | 0.71 | +0.17 |

**Catatan:** Data lengkap 3 musim. ✓

#### Aston Villa (Unai Emery):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 23/24 | 38 | 1.33 | 1.25 | +0.08 | 2.00 | 1.61 | +0.67 |
| 24/25 | 38 | 1.33 | 1.32 | +0.01 | 1.53 | 1.34 | +0.20 |
| 25/26 | 38 | 1.47 | 1.46 | +0.01 | 1.47 | 1.29 | 0.00 |

**Catatan:** Data lengkap 3 musim. ✓

#### Everton (David Moyes):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 24/25 | 38 | 1.17 | 1.40 | -0.23 | 1.11 | 1.16 | -0.06 |
| 25/26 | 38 | 1.31 | 1.55 | -0.24 | 1.24 | 1.32 | -0.07 |

**Catatan:** Data lengkap 2 musim. ✓

#### Brighton (Fabian Hurzeler):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 24/25 | 38 | 1.48 | 1.14 | +0.34 | 1.74 | 1.55 | +0.26 |
| 25/26 | 38 | 1.54 | 1.33 | +0.21 | 1.37 | 1.21 | -0.17 |

**Catatan:** Hurzeler mulai Brighton dari 24/25. Data lengkap 2 musim. ✓

#### Brentford (Keith Andrews — manajer baru tapi data tim berguna sebagai baseline):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 25/26 | 38 | 1.28 | 1.54 | -0.26 | 1.45 | 1.37 | +0.17 |

**Catatan:** Hanya 1 musim tersedia. Andrews dari Ireland NT — prior manajer tidak reliable, pakai data tim sebagai baseline. Data sangat terbatas ⚠

#### Sunderland (Regis Le Bris):
| Musim | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|----|----|-----|-----|----|----|--------------|
| 25/26 | 38 | 1.23 | 1.61 | -0.38 | 1.11 | 1.26 | -0.12 |

**Catatan:** Hanya 1 musim EPL tersedia (musim pertama). Data terbatas ⚠

---

### 11. Tim Promosi

#### Coventry City (Frank Lampard):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 25/26 | Championship | 46 | 1.83 | 1.27 | +0.56 | 2.11 | 0.98 | +0.28 |

**Weight Championship:** 0.3436 (estimasi)

#### Hull City (Sergej Jakirovic):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 25/26 | Championship | 49 | 1.28 | 1.63 | -0.35 | 1.49 | 1.35 | +0.21 |

**Weight Championship:** 0.3436 (estimasi)  
**Catatan:** MP=49 karena termasuk playoff Championship.

---

### 15. Alvaro Arbeloa → Fulham
**Prior klub:** Real Madrid Castilla (Primera Federación Tier 3) | **League weight:** 0.0589

| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 24/25 | Primera Federación | 38 | 1.49 | 1.23 | +0.26 | 1.53 | 0.95 | +0.04 |

**Catatan:** xG tersedia tapi league weight 0.0589 sangat kecil — kontribusi ke model ~6% saja di laga pertama, mendekati 0% setelah 15 laga karena digantikan data aktual Fulham. Prior ini ada tapi practically tidak berpengaruh ke prediksi akhir. Data 25/26 belum tersedia (Primera Federación baru mulai).

**delta_attack  (raw):** 1.49 - league_avg_xg → perlu league avg Primera Federación sebagai baseline  
**delta_defence (raw):** 1.23 - league_avg_xga → perlu league avg Primera Federación sebagai baseline  
**Rekomendasi:** Gunakan sebagai sinyal style (attacking intent ada, defence solid) bukan angka delta absolut karena liga berbeda level.

---

### 16. Prior Stints Manajer Lama (established managers' actual previous clubs)

Ini beda dari section #10 di atas — #10 isinya performa manajer di klub **sekarang** (established, dipakai buat konteks/laporan). Section ini isinya performa mereka di klub **sebelumnya** (dipakai `manager_prior()` buat blend, sesuai `config.MANAGERS[team]['prior_stints']`).

#### West Ham United (David Moyes, prior stint sebelum Everton):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 19/20 | EPL | 38 | 1.33 | 1.54 | -0.21 | 1.29 | 1.63 | -0.04 |
| 20/21 | EPL | 38 | 1.35 | 1.40 | -0.05 | 1.63 | 1.24 | +0.28 |
| 21/22 | EPL | 38 | 1.43 | 1.58 | -0.15 | 1.58 | 1.34 | +0.15 |
| 22/23 | EPL | 38 | 1.38 | 1.47 | -0.09 | 1.11 | 1.45 | -0.27 |
| 23/24 | EPL | 38 | 1.17 | 1.72 | -0.55 | 1.58 | 1.95 | +0.41 |

**Catatan:** Spell kedua Moyes (Des 2019-Mei 2024), spell pertama (Nov 2017-Mei 2018) tidak termasuk. Data lengkap 5 musim ✓

#### Chelsea (Frank Lampard, prior stint sebelum Coventry):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 19/20 | EPL | 38 | 1.86 | 1.05 | +0.81 | 1.82 | 1.42 | -0.04 |
| 20/21 | EPL | 38 | 1.71 | 1.05 | +0.66 | 1.53 | 0.95 | -0.18 |

**Catatan:** Jul 2019-Jan 2021. Data lengkap 2 musim ✓

#### Everton (Frank Lampard, prior stint sebelum Coventry):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 21/22 | EPL | 38 | 1.31 | 1.64 | -0.33 | 1.13 | 1.74 | -0.18 |
| 22/23 | EPL | 38 | 1.26 | 1.62 | -0.36 | 0.89 | 1.50 | -0.37 |

**Catatan:** Feb 2022-Jan 2023, keduanya partial. Data lengkap 2 musim ✓

#### Leicester City (Enzo Maresca, prior stint sebelum Chelsea):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 23/24 | Championship | 46 | 1.47 | 0.99 | +0.48 | 1.93 | 0.89 | +0.46 |

**Catatan:** Musim juara promosi. Data lengkap 1 musim ✓

#### Wolves (Gary O'Neil, prior stint sebelum Ipswich):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 23/24 | EPL | 38 | 1.14 | 1.52 | -0.38 | 1.32 | 1.71 | +0.18 |
| 24/25 | EPL | 38 | 1.21 | 1.38 | -0.17 | 1.42 | 1.82 | +0.21 |

**Catatan:** Aug 2023-Des 2024. Data lengkap 2 musim ✓

#### AFC Bournemouth (Gary O'Neil, prior stint sebelum Ipswich):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | EPL | 38 | 1.05 | 1.82 | -0.77 | 0.97 | 1.87 | -0.08 |

**Catatan:** ⚠️ Ini baris yang sama yang sempat salah nempel di prior Iraola (lihat koreksi di #1). Aug 2022-Jun 2023, satu-satunya musim penuh O'Neil di Bournemouth. Data lengkap 1 musim ✓

#### FC St. Pauli (Fabian Hürzeler, prior stint sebelum Brighton):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | 2. Bundesliga | 34 | 1.68 | 1.18 | +0.50 | 1.62 | 1.15 | -0.06 |
| 23/24 | 2. Bundesliga | 34 | 1.54 | 0.94 | +0.60 | 1.82 | 1.06 | +0.28 |

**Catatan:** Des 2022-Jun 2024. Data lengkap 2 musim ✓

#### FC Lorient (Régis Le Bris, prior stint sebelum Sunderland):
| Musim | Liga | MP | xG | xGA | xGD | GF | GA | xG vs Actual |
|-------|------|----|----|-----|-----|----|----|--------------|
| 22/23 | Ligue 1 | 38 | 1.20 | 1.69 | -0.49 | 1.37 | 1.39 | +0.17 |
| 23/24 | Ligue 1 | 34 | 1.15 | 1.61 | -0.46 | 1.26 | 1.94 | +0.11 |

**Catatan:** Musim relegasi 23/24. Data lengkap 2 musim ✓

---

## ❌ DATA xG YANG MASIH KURANG

| Prioritas | Manajer | Tim | Liga | Musim Dibutuhkan | Alasan |
|-----------|---------|-----|------|-----------------|--------|
| ~~🔴 TINGGI~~ | ~~Matthias Jaissle~~ | ~~RB Salzburg~~ | ~~Austrian BL~~ | ~~2021/22, 2022/23~~ | ✅ SKIP — data berbayar, pakai Al-Ahli only |
| ~~🔴 TINGGI~~ | ~~Roberto De Zerbi~~ | ~~Brighton~~ | ~~EPL~~ | ~~2022/23, 2023/24~~ | ✅ RESOLVED |
| ~~🔴 TINGGI~~ | ~~Oliver Glasner~~ | ~~Crystal Palace~~ | ~~EPL~~ | ~~2023/24~~ | ✅ RESOLVED |
| ~~🟡 SEDANG~~ | ~~Pierre Sage~~ | ~~Lyon~~ | ~~Ligue 1~~ | ~~23/24 partial~~ | ✅ RESOLVED (pakai 23/24 + 24/25, exclude 22/23 pre-Sage) |
| ~~🟡 SEDANG~~ | ~~Arsenal~~ | ~~Arsenal~~ | ~~EPL~~ | ~~23/24~~ | ✅ RESOLVED |
| ~~🟡 SEDANG~~ | ~~Michael Carrick~~ | ~~Middlesbrough~~ | ~~Championship~~ | ~~22/23-24/25~~ | ✅ RESOLVED — data Middlesbrough (klub lama asli), bukan Man Utd |
| ~~🟠 RENDAH~~ | ~~Alvaro Arbeloa~~ | ~~Real Madrid Castilla~~ | ~~Primera Fed~~ | ~~2024/25~~ | ✅ RESOLVED — xG 24/25 tersedia, weight tetap 0.0589 |
| ~~🟠 RENDAH~~ | ~~Gary O'Neil~~ | ~~Wolves + Bournemouth~~ | ~~EPL~~ | ~~23/24-24/25, 22/23~~ | ✅ RESOLVED |
| ~~🟠 RENDAH~~ | ~~David Moyes~~ | ~~West Ham~~ | ~~EPL~~ | ~~19/20-23/24~~ | ✅ RESOLVED |
| ~~🟠 RENDAH~~ | ~~Frank Lampard~~ | ~~Chelsea + Everton~~ | ~~EPL~~ | ~~19/20-20/21, 21/22-22/23~~ | ✅ RESOLVED |
| ~~🟠 RENDAH~~ | ~~Enzo Maresca~~ | ~~Leicester City~~ | ~~Championship~~ | ~~23/24~~ | ✅ RESOLVED |
| ~~🟠 RENDAH~~ | ~~Fabian Hürzeler~~ | ~~St. Pauli~~ | ~~2. Bundesliga~~ | ~~22/23-23/24~~ | ✅ RESOLVED |
| ~~🟠 RENDAH~~ | ~~Régis Le Bris~~ | ~~Lorient~~ | ~~Ligue 1~~ | ~~22/23-23/24~~ | ✅ RESOLVED |
| 🟠 RENDAH | Gary O'Neil | **Strasbourg** | Ligue 1 | Jan-Jun 2026 (partial) | Stint terbaru sebelum Ipswich, tapi W-D-L final gak ketemu bersih — di-skip di config.py, bukan ditebak |
| 🟠 RENDAH | Matthias Jaissle | **RB Salzburg** | Austrian BL | 2021/22, 2022/23 | Data berbayar, permanen skip |

---

## RINGKASAN STATUS DATA (update 2026-08)

```
✅ 20 dari 21 prior stint di manager_priors.json punya xG asli:
   Iraola (Bournemouth)        — 3 musim EPL (23/24-25/26, dikoreksi dari 4)
   Xabi Alonso (Leverkusen)    — 3 musim Bundesliga
   Marco Rose (Leipzig)        — 3 musim Bundesliga
   Maresca (Chelsea + Leicester) — 2 + 1 musim
   Jaissle (Al-Ahli)           — 3 musim Saudi PL (Salzburg di-skip)
   De Zerbi (Marseille + Brighton) — 2 + 4 musim
   Glasner (Crystal Palace)    — 3 musim EPL
   Carrick (Middlesbrough)     — 3 musim Championship
   Sage (Lyon)                 — 2 musim (23/24 partial + 24/25)
   Arbeloa (Real Madrid Castilla) — 1 musim
   Hurzeler (St. Pauli)        — 2 musim 2. Bundesliga
   Moyes (West Ham)            — 5 musim EPL
   Lampard (Chelsea + Everton) — 2 + 2 musim
   O'Neil (Wolves + Bournemouth) — 2 + 1 musim
   Le Bris (Lorient)           — 2 musim Ligue 1

❌ Belum ada (1 dari 21):
   Jaissle (Red Bull Salzburg) — data berbayar, permanen skip

Bug ketemu & dibenerin:
   Iraola/Bournemouth 22/23 salah nempel (itu musim O'Neil, Iraola baru
   masuk Jun 2023) — ketauan karena user kasih baris identik buat entri
   O'Neil terpisah. Dikoreksi di manager_priors.json + di sini.
```
