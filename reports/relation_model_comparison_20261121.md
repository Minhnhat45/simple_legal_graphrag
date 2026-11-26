# Đối chiếu Gemma 4 và GPT-6-Luna trong xây dựng quan hệ pháp luật

## Câu hỏi thực nghiệm

Nghiên cứu đo liệu thay Gemma 4 cục bộ bằng GPT-6-Luna có cải thiện chất lượng cạnh khi hai mô hình nhận cùng cặp điều khoản tiếng Việt. Ba phép đo xem xét chất lượng trích xuất, khả năng xác minh đề xuất và giá trị truy hồi của đồ thị làm giàu. Kết quả áp dụng cho hai cấu hình triển khai đã khóa, trong đó Gemma dùng Ollama và bản GGUF Q4_K_M, còn Luna dùng OpenRouter với reasoning medium. Phiên bản Luna được ghi theo canonical slug do nhà cung cấp báo về. Khoảng cách số tham số và ảnh hưởng riêng của kích thước mô hình chưa được xác định bằng thiết kế này. [Giao thức](runs/relation_model_comparison_20261121_v2/protocol.json) · [Bối cảnh hai lần build lịch sử](../docs/RESEARCH_EXPERIMENT_JOURNEY.md#66-đối-chiếu-gemma-4-cục-bộ-và-gpt-6-luna-khi-xây-dựng-quan-hệ)

## Dữ liệu và đối chứng

- **Cùng nguồn và cùng đầu vào.** Snapshot OpenRouter ngày 19/11 được mở chỉ đọc. Bộ dựng quét toàn văn điều khoản để đề xuất span trực tiếp có số điều và số hiệu đích, phân giải cả hai ID theo khóa duy nhất. Bộ dựng không đọc cạnh, kết quả của hai mô hình hay câu hỏi đánh giá retriever. Danh sách span được rà trước khi gọi mô hình. Với mỗi span giữ lại, tạo một cặp thật, một cặp đảo hướng và một cặp cùng văn bản đích nhưng thay bằng điều khác không xuất hiện trong đoạn nguồn. Tập đo là benchmark thăm dò dựa trên chứng cứ văn bản, chưa có hai chuyên gia pháp luật gán nhãn độc lập. [Bộ chọn đã lưu](relation_model_comparison_20261121_selection.json) · [Benchmark đã khóa](runs/relation_model_comparison_20261121_v2/benchmark.json)
- **Tách tập.** Development có 8 nhóm, 24 cặp. Test có 55 nhóm, 165 cặp. Hai nhóm dùng chung số hiệu được gộp cụm, và không có số hiệu giữa development/test. Một nhóm đề cử bị loại vì thiếu điều đối chứng cùng văn bản. Mỗi cặp chỉ cung cấp ID mờ, số điều, số hiệu và đoạn A/B. Gold, biến thể, span đánh giá và kết luận của mô hình kia không nằm trong request. `NONE` nghĩa là lớp quan hệ đề ra chưa có đủ căn cứ ở hướng A → B trong đoạn đã cung cấp. [File đầu vào và lần loại](runs/relation_model_comparison_20261121_v2/benchmark.json) · [Mã dựng](../scripts/run_relation_model_comparison.py)
- **Trích xuất và xác minh.** Hai mô hình nhận cùng nội dung, nhóm bốn cặp, ontology một nhãn ưu tiên theo hướng A → B, giới hạn 4.096 token đầu ra và schema logic chung. Sau kiểm tra schema/ID, dự đoán có kiểu phải có quote khớp liên tục với đoạn A sau NFC và chuẩn hóa khoảng trắng. Lần thử kỹ thuật của Gemma qua `/api/chat` sinh văn xuôi hoặc JSON lỗi. Giao thức cuối dùng `/api/generate` với định dạng Gemma 4, tiền tố kênh suy nghĩ rỗng và JSON schema. Cả hai nhánh nhận thêm cùng mẫu JSON. Tập test giữ nguyên và kết quả lượt kỹ thuật nằm riêng. Hai verifier xem cùng hợp các đề xuất đã qua xác thực, khử trùng theo cặp, nhãn và quote, không thấy mô hình đề xuất. [Hiện vật thử kỹ thuật](runs/relation_model_comparison_20261121/) · [Giao thức cuối](runs/relation_model_comparison_20261121_v2/protocol.json) · [Định dạng Gemma 4](https://gemma.ai/get-started/prompt-formatting/)
- **Chỉ số.** Chỉ số chính trên test là macro-F1 của `ADDS_TO`, `AMENDS`, `REPEALS` sau kiểm tra quote. Chỉ số phụ tính trên cả năm loại phi `CITES`, và phép chấm nghiêm ngặt yêu cầu quote chứa toàn span neo gold. `DETAILS` và `REPLACES` được báo riêng vì ít ca. Bootstrap ghép cặp 5.000 lần lấy lại cụm nối bởi số hiệu, seed 20261121. Một lần lấy mẫu thiếu nhóm dương của một nhãn tính macro trên các nhãn còn có support. Khác biệt thực tế tham khảo là 5 điểm F1, ghi trong giao thức trước khi chạy lần cuối. [Mã chấm và bootstrap](../scripts/run_relation_model_comparison.py)
- **Truy hồi có điều biết trước.** Tác vụ lấy các cặp phi `CITES` thật từ test, cung cấp điều nguồn và đoạn chứa dẫn chiếu, dùng cùng Qdrant BGE-M3/BM25 RRF, pool 60, tối đa 12 ứng viên follow-up, top 10 điều và ngân sách 18.000 ký tự. So sánh Hybrid, rule graph CITES từ snapshot Linux, rule graph cộng cạnh mỗi mô hình và lookup theo khóa. Cạnh LLM được giới hạn đúng đoạn và kiểu quan hệ cần tìm. Phép thử này đo đóng góp của cạnh trên cùng cặp ứng viên đã khảo sát trong một bài toán dẫn chiếu có neo. [Giao thức truy hồi](runs/relation_model_comparison_20261121_v2/retrieval/protocol.json) · [Mã truy hồi](../scripts/evaluate_relation_model_retrieval.py)

## Kết quả trích xuất trên 165 cặp test

Test có 55 cặp dương và 110 đối chứng âm. Ba lớp chính gồm 4 `ADDS_TO`, 6 `AMENDS` và 16 `REPEALS`. Phần còn lại có 1 `DETAILS`, 2 `REPLACES` và 26 `CITES`. Có 46 cụm lấy mẫu bootstrap. Development hoàn tất đủ 24/24 đầu ra hợp schema ở cả hai mô hình sau khi chuyển sang giao thức cuối. Các bảng dưới dùng toàn bộ test, kể cả dự đoán rớt xác thực. [Đầu ra development](runs/relation_model_comparison_20261121_v2/summary_development.json) · [Đầu ra test](runs/relation_model_comparison_20261121_v2/summary_test.json)

| Chỉ số | Gemma 4 | GPT-6-Luna |
| --- | ---: | ---: |
| Macro-F1 chính sau kiểm tra quote | **33,87%** | **100,00%** |
| Macro-F1 chính với yêu cầu chứa toàn span gold | 33,87% | 100,00% |
| Macro-F1 chính khi chỉ chấm nhãn, phân tích bổ sung | 45,92% | 100,00% |
| Macro-F1 của cả năm loại phi `CITES` | 30,32% | 93,33% |
| Micro-precision của năm loại phi `CITES` | 38,10% | 96,67% |
| Micro-recall của năm loại phi `CITES` | 55,17% | 100,00% |
| Micro-F1 của năm loại phi `CITES` | 45,07% | 98,31% |
| Đầu ra hợp schema và có ID duy nhất | 156/165 | 165/165 |
| Không qua schema/ID hoặc quy tắc quote | 77/165 | 0/165 |
| Đối chứng âm bị xuất cạnh sau kiểm tra quote | 34/110 | 1/110 |
| Đối chứng âm được trả `NONE` hợp lệ | 13/110 | 109/110 |

Chênh lệch chính **Luna − Gemma = +66,13 điểm phần trăm**. Khoảng tin cậy bootstrap ghép cặp 95% là **[54,42; 80,18] điểm**. Khi chỉ chấm nhãn từ mọi phản hồi hợp schema, vẫn tính các ca thiếu phản hồi hoặc lỗi schema là không có cạnh, chênh là **+54,08 [45,95; 66,52] điểm**. Phân tích bổ sung cho thấy khoảng cách còn lớn khi bỏ kiểm tra quote. [Tổng hợp máy đọc](relation_model_comparison_20261121.json) · [Kết quả và quote từng cặp](runs/relation_model_comparison_20261121_v2/scored_test.json)

| Loại cạnh | Số gold | Gemma TP / FP / FN | Gemma F1 | Luna TP / FP / FN | Luna F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ADDS_TO` | 4 | 1 / 5 / 3 | 20,00% | 4 / 0 / 0 | 100,00% |
| `AMENDS` | 6 | 1 / 4 / 5 | 18,18% | 6 / 0 / 0 | 100,00% |
| `REPEALS` | 16 | 13 / 12 / 3 | 63,41% | 16 / 0 / 0 | 100,00% |
| `DETAILS` | 1 | 1 / 2 / 0 | 50,00% | 1 / 1 / 0 | 66,67% |
| `REPLACES` | 2 | 0 / 3 / 2 | 0,00% | 2 / 0 / 0 | 100,00% |
| `CITES` | 26 | 19 / 13 / 7 | 65,52% | 25 / 1 / 1 | 96,15% |

Trong 77 ca không qua xác thực của Gemma, 9 ca thiếu đầu ra hợp schema/ID, 62 quote không xuất hiện liên tục trong đoạn A và sáu quote có trong A nhưng nằm ngoài giới hạn độ dài 12–350 ký tự. Sau bước lọc, Gemma vẫn sinh cạnh sai ở 31 cặp sai điều cùng số hiệu và ba cặp đảo chiều. Ở cặp `5ed3a136587d`, Gemma lấy cụm “Điều khoản được bãi bỏ” trong văn bản bị tác động làm bằng chứng cho hướng bãi bỏ ngược. Luna trả `NONE` đúng theo nhãn tham chiếu. Lỗi đáng chú ý của Luna ở `1ec21ddea546` là gán `DETAILS` cho đoạn “theo quy định tại …, cụ thể như sau” mà quy ước hẹp của benchmark xếp `CITES`. Đây là ca ranh giới cần chuyên gia phân xử khi mở rộng ontology. Ca `a109a6eca08c` cho thấy Luna vẫn có thể nối nhầm sang điều lân cận cùng số hiệu. [Dự đoán và chứng cứ](runs/relation_model_comparison_20261121_v2/scored_test.json)

## Xác minh chéo và đóng góp của từng công đoạn

Cả hai verifier trả đủ **127/127** phán quyết hợp schema trên cùng đề xuất. Tập này gồm 84 đề xuất đúng kiểu và chứa toàn span gold, cùng 43 đề xuất còn lại. Vì cùng một cặp có thể có hai quote khác nhau, đơn vị ở đây là đề xuất đã khử trùng theo cặp, nhãn và quote. Theo chuẩn đó, Luna giữ **79/84 (94,05%)** đề xuất được gold ủng hộ và loại **38/43 (88,37%)** đề xuất còn lại. Gemma giữ **68/84 (80,95%)** và loại **17/43 (39,53%)**. [Hợp đề xuất trước xác minh](runs/relation_model_comparison_20261121_v2/proposals_test.json) · [Tổng hợp xác minh](runs/relation_model_comparison_20261121_v2/summary_test.json)

| Mô hình đề xuất | Mô hình xác minh | Macro-F1 chính | Cạnh phi `CITES` đúng / gold | Cạnh sai trên 110 đối chứng âm |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 | Gemma 4 | 37,96% | 16/29 | 20 |
| Gemma 4 | GPT-6-Luna | 50,24% | 15/29 | 0 |
| GPT-6-Luna | Gemma 4 | 93,52% | 24/29 | 0 |
| GPT-6-Luna | GPT-6-Luna | **98,92%** | 28/29 | 1 |

Đổi cả hai công đoạn tăng macro-F1 chính **60,96 điểm [46,24; 78,76]**. Phân tích bổ sung cho thấy khi giữ verifier Gemma, đổi riêng mô hình đề xuất tăng **55,56 [32,98; 76,67]** điểm. Khi giữ proposer Gemma, đổi riêng verifier tăng **12,28 [3,87; 20,32]** điểm. Trên mẫu này, phần lớn chênh lệch gắn với khâu đề xuất. Verifier Luna loại nhiều cạnh sai hơn, nhưng có thể loại cả cạnh đúng. Cấu hình Luna hai công đoạn bỏ mất một cạnh gold ở test, làm macro-F1 chính giảm nhẹ so với bộ phân loại Luna trước xác minh. [Bảng 2 × 2 và so sánh ghép cặp](relation_model_comparison_20261121.json)

## Kết quả truy hồi

| Cấu hình | Đủ hai điều @10 | Điều đích @3 | Đích đúng trong danh sách ưu tiên trước Hybrid backfill |
| --- | ---: | ---: | ---: |
| Hybrid với cùng điều nguồn và đoạn trích | 18/29 (62,07%) | 8/29 (27,59%) | — |
| Rule graph + Hybrid backfill | 27/29 (93,10%) | 25/29 (86,21%) | 19/29 |
| Rule graph + cạnh Gemma/Gemma + backfill | **29/29 (100,00%)** | 27/29 (93,10%) | 24/29 |
| Rule graph + cạnh Luna/Luna + backfill | **29/29 (100,00%)** | **29/29 (100,00%)** | 29/29 |
| Tra chính xác theo khóa | **29/29 (100,00%)** | **29/29 (100,00%)** | 29/29 |

Cả hai đồ thị làm giàu thu lại hai ca thiếu của rule graph trong top 10. Mỗi nhánh tăng **6,90 [0,00; 18,52]** điểm so với rule graph. Chênh Luna − Gemma ở chỉ số chính bằng **0,00 điểm** vì hai nhánh giữ đủ gold ở cả 29 câu. Luna có thêm hai đích ở top 3 và thêm năm đích trong danh sách ưu tiên. Đây là kết quả phụ trên một tác vụ hẹp. Danh sách ưu tiên của hai nhánh gồm cả cạnh LLM và CITES theo quy tắc, nên 29/29 của Luna ở cột cuối có thể gồm cạnh từ nhánh quy tắc. Đối chứng exact key cũng đạt 29/29. Khi đoạn nguồn có định danh đủ, các đối chứng có thể bù cho cạnh LLM bị thiếu, làm chỉ số đủ gold bão hòa dù chất lượng đồ thị khác nhau. [Tổng hợp truy hồi](runs/relation_model_comparison_20261121_v2/retrieval/summary.json) · [Trace từng câu](runs/relation_model_comparison_20261121_v2/retrieval/rows.json)

## Thời gian, chi phí và khả năng tái lập

Lần chạy cuối lưu **86 yêu cầu Luna**, **139.074 prompt token**, **32.482 completion token** và chi phí API ghi nhận **0,033575700 USD** cho cả development/test và hai công đoạn. Lượt thăm dò kỹ thuật trước đó lưu 48 yêu cầu Luna và **0,018947375 USD**. Tổng hai lượt là **0,052523075 USD** theo usage đã lưu. Trung vị thời gian trả về một batch phân loại test là **5,03 s** cho Gemma và **4,05 s** cho Luna. Đây là số đo của GPU tại chỗ và API từ xa với mức đồng thời khác nhau. Chi phí điện, phần cứng và so sánh throughput dưới cùng tải chưa được đo. [Chi phí và thời gian](relation_model_comparison_20261121.json) · [Giao thức mô hình](runs/relation_model_comparison_20261121_v2/protocol.json)

Mã mới và dữ liệu được tách khỏi pipeline chính. Chạy từ thư mục repo với snapshot có sẵn và môi trường `.venv-linux`. Sao chép benchmark đã khóa nếu cần đối chiếu byte với giao thức đã công bố. Chế độ `prepare` từ danh sách chọn tái tạo đúng từng cặp và trường hợp bị loại, nhưng có thể ghi hash phiên bản builder khác. Khi chạy lại với thư mục mới, các bước là `preflight`, `classification` cho hai mô hình, `proposals`, `verification` cho hai mô hình, rồi `summarize`. Các lệnh nhận `--output`, `--split` và `--model` ở bước tương ứng. Runner truy hồi nhận `prepare` và `evaluate` với `--run` và `--output`. Các hiện vật dưới `reports/runs/` cần được sao chép kèm. [Runner mô hình](../scripts/run_relation_model_comparison.py) · [Runner truy hồi](../scripts/evaluate_relation_model_retrieval.py) · [Mã đã chạy và đầu ra](runs/relation_model_comparison_20261121_v2/)

## Kết luận dùng trong luận văn

Trên **tập cặp điều khoản có quan hệ biểu đạt trực tiếp trong văn bản**, GPT-6-Luna qua OpenRouter trích xuất quan hệ có hướng và dẫn đúng đoạn tốt hơn cấu hình Gemma 4 cục bộ đã thử. Kết quả vẫn giữ khi chỉ so nhãn, và thiết kế xác minh chéo cho thấy lợi thế lớn hơn ở khâu đề xuất. Phép thử truy hồi xác nhận cả hai đồ thị làm giàu đạt đủ hai điều trên toàn bộ 29 câu. Cấu hình Luna đưa đích lên top 3 thường xuyên hơn trên mẫu này, còn ưu thế về đủ gold ở top 10 chưa xuất hiện.

Kết quả là bằng chứng về **hai lựa chọn triển khai**. Muốn lập luận rằng *tăng kích thước mô hình* tạo ra lợi ích, cần giữ họ mô hình, dữ liệu đầu vào, template, mức suy luận và mức lượng tử hóa tương đương khi so các biến thể dung lượng khác nhau. Phép thử hiện tại ưu tiên quan hệ quy phạm có dấu hiệu tường minh, dùng đối chứng âm tạo từ điều lân cận và đảo hướng, và chưa có gold độc lập do chuyên gia luật rà soát. Do đó, số đo chưa khái quát cho các kiểu quan hệ ngữ nghĩa ngầm, tính hiệu lực pháp lý hiện hành hoặc truy hồi tự nhiên cần suy luận nhiều bước.
