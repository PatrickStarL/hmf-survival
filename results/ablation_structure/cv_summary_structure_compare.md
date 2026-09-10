# P1 Fusion Structure Compare Summary

- Done runs: 135 / 165
- Failed runs: 29
- Skipped runs: 1
- Overall mean c-index(test): 0.6728584012472744
- Overall std c-index(test): 0.10168057147576941

## By Structure

| Structure | Done | Fail | Skip | Mean c-index(test) | Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| coattn_text | 45 | 0 | 0 | 0.6846143970077044 | 0.09290228570986717 |
| shallow_cls | 45 | 29 | 1 | 0.6576115209022589 | 0.10906294840769024 |
| twostage | 45 | 0 | 0 | 0.6763492858318598 | 0.10054052600649897 |

## By Structure + Dataset

| Structure | Dataset | Done | Fail | Skip | Mean c-index(test) | Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coattn_text | CRC | 15 | 0 | 0 | 0.7867006797956396 | 0.062077617627129676 |
| coattn_text | LUAD | 15 | 0 | 0 | 0.6309471061454711 | 0.06266327556200055 |
| coattn_text | STAD | 15 | 0 | 0 | 0.6361954050820026 | 0.04965884951227751 |
| shallow_cls | CRC | 15 | 0 | 0 | 0.7666912384471262 | 0.0672127802656382 |
| shallow_cls | LUAD | 15 | 15 | 0 | 0.6215067281810506 | 0.06806002183301019 |
| shallow_cls | STAD | 15 | 14 | 1 | 0.5846365960786 | 0.08948282099409137 |
| twostage | CRC | 15 | 0 | 0 | 0.7781150664077333 | 0.0823939991611675 |
| twostage | LUAD | 15 | 0 | 0 | 0.6495798587251309 | 0.05732648489692472 |
| twostage | STAD | 15 | 0 | 0 | 0.6013529323627154 | 0.05960508185611803 |
