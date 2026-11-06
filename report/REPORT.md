# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (CUDA - T4)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   **Trả lời:** Record này nói về **cả bức ảnh**. Vì đây là tác vụ phân loại ảnh (image classification), mô hình dự đoán và gán một nhãn phân loại tổng thể cho toàn bộ bức ảnh chứ không xác định vị trí hay phân tách từng vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   **Trả lời:** Danh sách tên lớp **đi kèm theo taxonomy/checkpoint** (cụ thể ở đây là taxonomy ImageNet-1K đi kèm checkpoint `yolo11n-cls.pt`), mô hình không thể tự nghĩ ra lớp mới nằm ngoài taxonomy đã được huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   **Trả lời:** Cần ghi cả 3 trường vì `class_name` có thể mơ hồ hoặc trùng lặp giữa các bộ dữ liệu khác nhau; `class_id` giúp định danh duy nhất và lập chỉ mục chính xác trong hệ thống mã nguồn; `taxonomy_name` xác định rõ không gian nhãn và quy chuẩn phân loại đang áp dụng (ví dụ: ImageNet-1K khác với COCO-80).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   **Trả lời:** Guideline cần quy định rõ nguyên tắc ưu tiên khi ảnh chứa nhiều đối tượng: chọn đối tượng chiếm diện tích chủ đạo/trọng tâm ảnh, hoặc chọn lớp theo mục tiêu nghiệp vụ cụ thể của hệ thống (ví dụ: ưu tiên phương tiện giao thông hơn bối cảnh đường phố).
6. Vì sao không được lấy `score` của model làm ground truth?  
   **Trả lời:** Vì `score` chỉ là xác suất/độ tin cậy toán học của mô hình đưa ra dựa trên trọng số đã học, hoàn toàn có thể bị sai lệch (false positive hoặc hallucination); ground truth bắt buộc phải là nhãn do con người thẩm định và xác nhận theo quy tắc nghiệp vụ chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   **Trả lời:** Bounding box có góc trên-trái tại pixel (x=385.33, y=69.24) tính từ gốc tọa độ (0, 0) ở góc trên bên trái của ảnh; góc dưới-phải tại pixel (x=498.92, y=348.92); kích thước hộp có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   **Trả lời:**  
   - Ngưỡng `0.20`: có **17** prediction (gồm person, bowl, oven, cup, spoon, potted plant, dining table, bottle).  
   - Ngưỡng `0.60`: có **6** prediction (chỉ còn person, bowl, oven).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   **Trả lời:** Khi hạ ngưỡng, số lượng bounding box dự đoán tăng lên nhiều (từ 6 lên 17 hộp, xuất hiện thêm nhiều đối tượng có độ tin cậy thấp và dự đoán sai lệch/nhiễu). Người reviewer phải tốn nhiều thời gian hơn để kiểm tra từng hộp, loại bỏ các hộp phát hiện sai (false positive), lọc trùng lặp và xác nhận lại ranh giới thật của vật thể.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   **Trả lời:** Vẽ bounding box bao khít toàn bộ phần nhìn thấy được của vật thể; không bao gồm bóng đổ hoặc hình phản chiếu; đối với vật thể bị che khuất một phần, chỉ đóng khung phần diện tích thực tế hiển thị nếu phần đó vẫn nhận diện rõ ràng (>20% diện tích).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   **Trả lời:** Trên ảnh `kitchen`, người phụ nữ thứ hai ở cạnh trái ảnh chỉ nhìn thấy một phần cánh tay/thân (`person` tại box [0.12, 263.18, 61.55, 310.78]) bị cắt mép ảnh nghiêm trọng và các đồ vật như bát/thìa xếp chồng lên nhau. Trường hợp vật bị cắt mép sát biên hoặc che khuất phần lớn này cần **escalation (hỏi Lab Coach)** để thống nhất quy chuẩn (ví dụ: vật bị cắt mép còn dưới 30% có gán nhãn không). Sau khi được Lab Coach hướng dẫn, quy tắc sẽ được chuẩn hóa vào guideline để toàn bộ annotator làm theo.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-010 ; `class_name`=bowl ; `score`=0.372406 ; số điểm polygon=`polygon_point_count`=12 ; 3–5 điểm đầu của `polygon_xy`=[[155.0, 170.0], [155.0, 177.0], [156.0, 178.0], [156.0, 179.0], [161.0, 184.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   **Trả lời:** Polygon cung cấp thông tin chi tiết về đường biên viền hình học thực tế, diện tích pixel thực và độ cong uốn lượn của vật thể; giúp tách biệt hoàn toàn phần thân của vật thể ra khỏi nền hoặc các vật thể khác xung quanh mà bounding box hình chữ nhật không làm được.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   **Trả lời:** `instance_id` dùng để định danh riêng cho từng cá thể vật thể độc lập trong cùng một bức ảnh (ví dụ: phân biệt cái bát `kitchen-010` với các cái bát khác trong cùng gian bếp). Nó không phải `class_id` vì nó không đại diện cho chủng loại; nó cũng không phải `tracking_id` vì chỉ tồn tại trong ảnh tĩnh này, không dùng để theo dõi hành trình chuyển động qua các khung hình theo thời gian.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   **Trả lời:** Vẽ các điểm polygon bám sát đường viền nhìn thấy của đối tượng với sai số không quá 2-3 pixel; đi theo ranh giới thực giữa vật thể và nền, không vẽ trùm ra bóng đổ và tách biệt ranh giới rõ ràng tại điểm tiếp xúc giữa hai vật thể liền kề.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   **Trả lời:** Tại vị trí các chiếc thìa cắm trong cốc hoặc bát xếp chồng chạm vào nhau tạo ra ranh giới tiếp xúc rất khó phân định, ta chọn **escalation** để hỏi Lab Coach thống nhất tiêu chuẩn tách hay gộp mask; sau đó mới đưa vào guideline chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp (`class_id`, `class_name`) cho toàn bộ bức ảnh | Ảnh `traffic` có nhiều phương tiện hỗn hợp nhưng model chỉ gán lớp `cab` (score ~0.51) | Chọn 1 lớp đại diện phù hợp nhất theo thứ tự ưu tiên trong guideline | Kiểm tra nhãn có đúng taxonomy và tuân thủ thứ tự ưu tiên khi có nhiều vật |
| Phát hiện vật thể | Danh sách các bounding box `[x_min, y_min, x_max, y_max]` kèm nhãn lớp cho từng vật | Các vật thể nhỏ hoặc bị khuất ở hậu cảnh bếp dễ bị bỏ sót (false negative) hoặc box bị thừa viền | Xác định vị trí, vẽ box bao sát từng vật thể đúng theo danh mục lớp | Soát xem box có bị lệch viền, thiếu vật thể hoặc bắt nhầm vật thể không |
| Instance segmentation | Tập hợp các đa giác kín `polygon_xy` kèm nhãn lớp cho từng cá thể | Ranh giới giữa các đồ vật liền kề hoặc chỗ đổ bóng dễ bị mô hình gộp nhầm hoặc cắt lẹm viền | Chấm các điểm polygon bám khít đường viền từng cá thể riêng biệt | Kiểm tra độ mịn và độ chính xác của đường biên mask, đảm bảo tách đúng instance |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   **Trả lời:** Chỉ sử dụng tập ảnh mẫu COCO công khai do bài lab cung cấp, tuyệt đối không tải lên Colab hay GitHub các hình ảnh cá nhân, biển số xe, khuôn mặt hoặc dữ liệu nội bộ.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   **Trả lời:** Tôi sẽ dừng ngay lập tức và báo cáo cho Giảng viên / Lab Coach phụ trách lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
