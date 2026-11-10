# Failure analysis of the 2026-11-07 frozen-seed experiment

Analysis date: 2026-11-11. Sources: frozen M0–M4 predictions, saved manifest, original eval.csv and corpus.csv, and local source coverage. The eval.csv SHA-256 matches the experiment. No retrieval was rerun or gold labels changed.

**The official `single_chunk` group fails most often:** 18/50 queries (36%) reach no target law, versus 7/50 (14%) for `multiple_chunks`. Exact-article misses are 25/50 (50%) and 9/50 (18%). These labels describe gold chunk multiplicity, not semantic complexity or multi-hop reasoning.

Law-level success means that an article from the exact corpus `law_id` appears among the saved top-10 articles. This is not a rerun retrieving ten distinct laws. A law hit can still be the wrong article. Shared law numbers can also be ambiguous across authorities/versions.

| Mode | Queries | No gold article | No target law | Missing ≥1 target law |
| --- | --- | --- | --- | --- |
| M0 | 100 | 34 | 25 | 27 |
| M1 | 100 | 34 | 25 | 27 |
| M2 | 100 | 34 | 25 | 27 |
| M3 | 100 | 34 | 25 | 27 |
| M4 | 100 | 34 | 25 | 27 |

All five modes miss exactly the same articles and laws: True. Of 34 zero-article-hit queries, nine nevertheless reach a target law. There are 25 complete law misses and two partial multi-law misses (10419 and 7932), for 27 affected queries and 28 missed query–law pairs.

**Coverage explains most of the gap.** Among queries with all target laws indexed, failures fall to 3/35 (8.6%) for `single_chunk` and 3/45 (6.7%) for `multiple_chunks`. For fully article-covered queries, zero-article-hit failures are 2/27 (7.4%) and 2/41 (4.9%). The raw group gap therefore does not establish that single-chunk wording is intrinsically harder.

| Stage of missed target law | Query–law pairs |
| --- | --- |
| no_local_source | 11 |
| law_candidate_outside_top_k | 2 |
| law_not_indexed | 9 |
| law_not_in_candidates | 6 |

Twenty of 28 missed query–law pairs have no indexed article from that law: 11 lack local source documents (metadata only), and nine have local documents but no mapped article in the pilot index. Six indexed laws never enter the candidate set, and two enter it but fall outside the final top 10. These are observable pipeline stages, not proven semantic root causes.

At article level, 43 gold article occurrences are missed: 36 are unindexed, five indexed articles are absent from candidates, and two are candidates excluded from the top 10. Thirty of the 34 zero-article-hit queries have no indexed gold article; four are fully covered retrieval/ranking failures.

**A concentrated wording-defined failure family is organization structure/functions:** 6/7 queries (85.7%) miss the target law. All six failed laws are absent from the index; five lack local source documents, while `12/2012/QH13` has a local source. The one success is qid 15823, targeting `61/2022/NĐ-CP`. This exploratory category is not supplied by the benchmark and has a small sample.

Tag rule (case-insensitive substring regex): `vị trí và chức năng|cơ cấu tổ chức|hệ thống tổ chức|trực thuộc|nhiệm vụ và quyền hạn`.

| qid | Question | Target law_id | Missing law? |
| --- | --- | --- | --- |
| 15823 | Vị trí và chức năng của Ban Quản lý Lăng Chủ tịch Hồ Chí Minh? | 61/2022/NĐ-CP | False |
| 15874 | Nhiệm vụ và quyền hạn của Báo Văn hóa được quy định như thế nào? | 759/QĐ-BVHTTDL | True |
| 15872 | Vị trí và chức năng của Cục Hợp tác quốc tế thuộc Bộ Văn hóa - Thể thao và Du lịch là gì? | 763/QĐ-BVHTTDL | True |
| 5950 | Cục Xúc tiến thương mại là cơ quan trực thuộc Bộ nào? | 963/QĐ-BCT | True |
| 16203 | Vị trí và chức năng của Cục Công nghiệp công nghệ thông tin và Truyền thông được quy định như thế nào? | 1816/QĐ-BTTTT | True |
| 3849 | Hệ thống tổ chức công đoàn bao gồm những đơn vị nào? | 12/2012/QH13 | True |
| 7642 | Cơ cấu tổ chức của Vụ Truyền thông và Thi đua khen thưởng Bộ Y tế? | 468/QĐ-BYT | True |

**Highest-frequency missed law:** `02/2022/TT-BTNMT`, missed in both of its two target queries: 7105 (publication of the environmental appraisal council list) and 14907 (consultation deadline for environmental impact assessment approval). Both are coverage failures: metadata exists, but no local source document or indexed article. Every other missed law occurs in only one failed query.

**Four fully covered complete misses to debug first:**

| qid | Question | Gold aid → law_id | Observed stage | Top-3 retrieved law_ids |
| --- | --- | --- | --- | --- |
| 6920 | Có bắt buộc phải về quê xin Giấy xác nhận tình trạng hôn nhân sau khi nhập hộ khẩu ở địa chỉ mới không? | 5469 → 23/2015/NĐ-CP | law_not_in_candidates | 82/2020/NĐ-CP, 55/2021/TT-BCA, 62/2021/NĐ-CP |
| 1164 | Nguyên tắc áp dụng định mức kinh tế - kỹ thuật cho lập và điều chỉnh quy hoạch quảng cáo ngoài trời được quy định như thế nào? | 8863 → 04/2023/TT-BVHTTDL | law_candidate_outside_top_k | 16/2012/QH13, 181/2013/NĐ-CP, 01/2023/NĐ-CP |
| 5124 | Cơ quan, đơn vị tiếp nhận động vật rừng do tổ chức, cá nhân tự nguyện giao nộp Nhà nước được quy định như thế nào? | 59320 → 22/2023/TT-BNNPTNT | law_not_in_candidates | 16/2017/QH14, 85/2023/NĐ-CP, 16/2017/QH14 |
| 16281 | Nhờ Ban tư vấn giải đáp giúp em vấn đề sau trong thời gian sớm nhất. Thắc mắc của em cụ thể như sau: Tiêu chuẩn đơn vị hành chính cấp xã được quy định như thế nào? | 46664 → 1211/2016/UBTVQH13 | law_not_in_candidates | 33/2023/NĐ-CP, 10/2015/NĐ-CP, 158/2018/NĐ-CP |

The returned laws suggest topic overlap without the required provision: 6920 drifts toward residence/certification topics; 5124 toward forestry and unrelated personnel rules; 16281 toward commune personnel rather than administrative-unit standards. Query 1164 reaches its exact gold candidate but loses it during ranking. These explanations are inspection-based hypotheses; the saved trace establishes candidate inclusion/exclusion but does not include component scores sufficient to prove why ranking failed.

Additional indexed-law misses: 4190 has the correct law among candidates but loses it at top 10, while its gold article is unindexed; 2519 and the missing `96/2023/NĐ-CP` portion of 10419 have indexed laws but unindexed gold articles and no target-law candidate. Query 2217 combines an indexed gold article (`17419`, `18/2022/TT-BYT`) absent from candidates with an entirely unindexed law (`56/2017/TT-BYT`). It is excluded from the fully covered subset but still exposes a real candidate-retrieval failure.

Two fully covered partial article failures already reach the correct law: 4852 misses aid `18207` before candidate ranking; 9882 includes aid `56079` among candidates but excludes it from the final top 10. Neither should be counted as failure to reach a law_id.

**All queries missing at least one target law (M4; identical missing-law sets in M0–M3):**

| qid | Group | Question | Missing law_id → stage |
| --- | --- | --- | --- |
| 7105 | single_chunk | Danh sách hội đồng thẩm định báo cáo đánh giá tác động môi trường có cần phải công khai không? | 02/2022/TT-BTNMT → no_local_source |
| 4190 | single_chunk | Mã số các ngạch công chức ngành nông nghiệp và phát triển nông thôn được quy định thế nào? | 08/2022/TT-BNNPTNT → law_candidate_outside_top_k |
| 13056 | single_chunk | Những nguyên tắc cần phải tuân thủ khi tổ chức đám cưới là gì? | 04/2011/TT-BVHTTDL → law_not_indexed |
| 10419 | multiple_chunks | Các trường hợp nào phải có giấy phép nhập khẩu trang thiết bị y tế? | 96/2023/NĐ-CP → law_not_in_candidates |
| 2519 | multiple_chunks | Giấy chứng nhận kiểm định xe cơ giới hết hiệu lực trong trường hợp nào? | 16/2021/TT-BGTVT → law_not_in_candidates |
| 15874 | multiple_chunks | Nhiệm vụ và quyền hạn của Báo Văn hóa được quy định như thế nào? | 759/QĐ-BVHTTDL → no_local_source |
| 15872 | single_chunk | Vị trí và chức năng của Cục Hợp tác quốc tế thuộc Bộ Văn hóa - Thể thao và Du lịch là gì? | 763/QĐ-BVHTTDL → no_local_source |
| 12024 | single_chunk | Chính sách trợ cấp tham quan, học tập, bồi dưỡng chuyên môn, nghiệp vụ đối với nhà giáo, cán bộ quản lý giáo dục công tác ở trường chuyên biệt được quy định như thế nào? | 61/2006/NĐ-CP → law_not_indexed |
| 11307 | single_chunk | Ban tư vấn cho tôi hỏi mức phí đánh giá lần đầu để cấp giấy chứng nhận quốc tế về an ninh tàu biển là bao nhiêu? | 246/2016/TT-BTC → law_not_indexed |
| 6735 | single_chunk | Ai là người có trách nhiệm phải nộp thuế tiêu thụ đặc biệt? | 27/2008/QH12 → law_not_indexed |
| 5950 | single_chunk | Cục Xúc tiến thương mại là cơ quan trực thuộc Bộ nào? | 963/QĐ-BCT → no_local_source |
| 16203 | single_chunk | Vị trí và chức năng của Cục Công nghiệp công nghệ thông tin và Truyền thông được quy định như thế nào? | 1816/QĐ-BTTTT → no_local_source |
| 10036 | single_chunk | Tiếp nhận, quản lý, sử dụng kinh phí đối với lễ hội không phải do cơ quan nhà nước tổ chức ra sao? | 04/2023/TT-BTC → no_local_source |
| 8289 | multiple_chunks | 1. Viên chức nghiện ma túy có bị buộc thôi việc hay không? | 112/2020/NĐ-CP → no_local_source |
| 16576 | single_chunk | Tôi có biết Bộ Tư pháp vẫn thường có các buổi gặp mặt cũng như lắng nghe ý kiến từ người dân, nhưng không rõ trách nhiệm tiếp công dân của Bộ Tư pháp được quy định như thế nào? | 266/QĐ-BTP → no_local_source |
| 6920 | single_chunk | Có bắt buộc phải về quê xin Giấy xác nhận tình trạng hôn nhân sau khi nhập hộ khẩu ở địa chỉ mới không? | 23/2015/NĐ-CP → law_not_in_candidates |
| 1164 | multiple_chunks | Nguyên tắc áp dụng định mức kinh tế - kỹ thuật cho lập và điều chỉnh quy hoạch quảng cáo ngoài trời được quy định như thế nào? | 04/2023/TT-BVHTTDL → law_candidate_outside_top_k |
| 14907 | single_chunk | Thời hạn lấy ý kiến phê duyệt kết quả thẩm định báo cáo đánh giá tác động môi trường đối với dự án đầu tư có hoạt động xả nước thải vào công trình thủy lợi? | 02/2022/TT-BTNMT → no_local_source |
| 5124 | single_chunk | Cơ quan, đơn vị tiếp nhận động vật rừng do tổ chức, cá nhân tự nguyện giao nộp Nhà nước được quy định như thế nào? | 22/2023/TT-BNNPTNT → law_not_in_candidates |
| 16281 | multiple_chunks | Nhờ Ban tư vấn giải đáp giúp em vấn đề sau trong thời gian sớm nhất. Thắc mắc của em cụ thể như sau: Tiêu chuẩn đơn vị hành chính cấp xã được quy định như thế nào? | 1211/2016/UBTVQH13 → law_not_in_candidates |
| 3849 | single_chunk | Hệ thống tổ chức công đoàn bao gồm những đơn vị nào? | 12/2012/QH13 → law_not_indexed |
| 11605 | single_chunk | 2. Báo cáo tài chính năm doanh nghiệp cần trình bày những thông tin nào? | 200/2014/TT-BTC → law_not_indexed |
| 7642 | multiple_chunks | Cơ cấu tổ chức của Vụ Truyền thông và Thi đua khen thưởng Bộ Y tế? | 468/QĐ-BYT → no_local_source |
| 4270 | single_chunk | Sinh con thứ 3, công chức có bị kéo dài thời hạn nâng lương? | 03/2021/TT-BNV → law_not_indexed |
| 3965 | single_chunk | Quân nhân chuyên nghiệp trong trường hợp nào không được hưởng trợ cấp một lần khi nghỉ hưu trước hạn tuổi? | 162/2017/TT-BQP → law_not_indexed |
| 7932 | multiple_chunks | Điều kiện đăng ký dự tuyển viên chức được quy định như thế nào? | 52/2019/QH14 → no_local_source |
| 2217 | multiple_chunks | Mắc các bệnh nào được hưởng chế độ bảo hiểm xã hội một lần theo quy định hiện nay? | 18/2022/TT-BYT → law_not_in_candidates; 56/2017/TT-BYT → law_not_indexed |

Recommended order: restore missing sources and index the current corpus more completely; inspect candidate generation for 6920, 5124, 16281 and the indexed part of 2217; inspect final ranking for 1164 and 4190 (and article completeness for 9882). Validate questionable gold mappings before changing them. Test any fixes on separate development/held-out data rather than tuning on these 100 questions. The graph modes recover no missed target law in this snapshot.

Reproduce with `python3 scripts/analyze_failures.py --output-prefix reports/failure_analysis_NEW`. The sibling project supplies eval.csv and corpus.csv; override their paths if needed. JSON contains input hashes, metric definitions, group summaries and all 500 query-mode records; CSV contains the same detailed records. The original corpus hash was not recorded by the experiment, so this analysis records its current hash and checks uniqueness and full referenced-ID resolution. Law metadata was joined from that corpus; legal correctness/current applicability was not independently assessed.
