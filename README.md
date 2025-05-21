# gait-recognition
Gait Recognition based on GaitGraph wth ST-GCN structure 
## Hướng dẫn sử dụng code 
### Lưu ý: Khi clone hết repository này về , hãy lưu tất cả repository này bên trong một thư mục đặt tên là gait_recognition và hãy chạy trên Google Colab để tránh lỗi
* Bước 1: Vào link https://drive.google.com/drive/folders/1dpcgN3qg463hdLKA_yVWnmHuiFv8UMeX?usp=sharing, tải file train.csv và test.csv
* Bước 2: Bên trong file GAIT.ipynb, đổi đường dẫn path đến 2 file vừa tải ở 2 biến toàn cục là train_file_path và test_file_path, bạn sẽ thấy 2 biến này ở ngay đoạn đầu file main , sau phần import thư viện.
  ![image](https://github.com/user-attachments/assets/9cca2cc1-d657-47b8-a56a-f6975043fdda)
  
* Bước 3: Bên trong file GAIT.ipynb, lướt xuống mục các hàm cơ bản và hàm train , valid. Bạn hãy đổi đường dẫn lưu model ở biến save_path nằm trong hàm fit_one_cycle() bằng đường dẫn lưu model và tên model sẽ được lưu.
  ![image](https://github.com/user-attachments/assets/8abce528-3faf-4af5-8ed1-e67c5507edb9)

* Còn nếu không muốn train lại từ đầu, bạn có thể tải model đã được train sẵn của chúng tôi ở link https://drive.google.com/drive/folders/1WbUY0OsZupK7LOFZoYiFlykIJ6k-dEoC?usp=sharing, tải file stgcn_model_2.pth.
* Bước 4: Load model tốt nhất lưu được và đánh giá nó. Thay đường dẫn đến model mà bạn đã thay đổi ở biến save_path vào biến checkpoint để load lại model vào.
  ![image](https://github.com/user-attachments/assets/d32f914a-d818-499c-86a8-3719bcc4b936)
  
  Hoặc đổi đường dẫn checkpoint đến đường dẫn mà bạn đã lưu model pretrained mà bạn đã tải.
* Nếu bạn muốn visualize các feature vector của các điểm dữ liệu trên tập train hay tập test thì bạn có thể uncomment các đoạn code mà bọn tôi đã comment để tránh việc lặp lại. Các đoạn code visualize feature vector có thể tìm thấy ngay sau khi evaluate model.
  ![image](https://github.com/user-attachments/assets/c6f0dc8b-808c-4820-bd73-e0fa4945fe37)
  uncomment this
  ![image](https://github.com/user-attachments/assets/077d5479-4a41-4c5d-a76e-e9520a70ddba)
  and uncomment this
  
  Bạn cũng có thể thay đổi đường dẫn lưu train_embedding và test_embedding tùy ý. Sau đó khi load lại file csv bạn chỉ cần đưa đường dẫn đã lưu 2 file đó là được.
  ![image](https://github.com/user-attachments/assets/e231f1c0-f5c1-438c-9722-15e385b4ad60)
  
### Về việc thêm dữ liệu ID người khác từ bên ngoài vào dataset và nhận dạng
* Bước 1: Bạn cần dữ liệu đầu vào là các video chứa người đang đi bộ , video cần phải chỉ chứ một người duy nhất. Sau đó bạn cần phải tách frame và lưu vào thư mục có định dạng tên như sau
  ![image](https://github.com/user-attachments/assets/e801631b-a6c6-48eb-b4c4-ae96f8e6cef6)
  
  Sau đó, bạn sẽ phải tạo một file csv chứa tên của toàn bộ frame để đồng bộ cho quá trình detector sắp tới.
  Nếu bạn chỉ có 1 video cho 1 ID người mời ở góc quay 0 độ thì bạn có thê chạy 3 cell như trong file main
  ![image](https://github.com/user-attachments/assets/22911cbe-313a-48ee-9f5b-39f582fdb7db)
  ![image](https://github.com/user-attachments/assets/44c249e6-f6fb-4cfd-aa07-ef033f1abbd6)
  Chỉ cần thay đổi đường dẫn đến file video và đường dẫn lưu folder mà bạn muốn.
  Tuy nhiên: Nếu bạn có nhiều video của ID người đó với nhiều góc độ khác nhau và nhiều loại gait type khác nhau, bạn sẽ phải lặp lại 3 thao tác này vài lần để lưu toàn bộ frame vào các folder riêng của nó, lưu nó thành một folder lớn làm data và một file images_name.csv chứa toàn bộ tên của tất cả frame.(Khuyến khích bạn có nhiều dữ liệu)
* Bước 2: Bạn vào đường link này https://drive.google.com/drive/folders/1gJ8T7c9Q89aXsusZf1lrF1xhNcCGMd42?usp=sharing, tải 3 pretrained model bên trong về và lưu toàn bộ vào một thư mục đặt tên là models ở bên trong thư mục gốc gait_recognition.
* Bước 3: Chạy các đoạn lệnh sau
  ![image](https://github.com/user-attachments/assets/09aaa400-ddc5-4578-ada7-7ba3f3912d90)
  Nó gồm 4 arguments: argument đầu tiên là đường dẫn đến file prepare_detection.py (bạn có thể tìm thấy file này bên trong thư mục HRNet->Detector), argument thứ hai là đường dẫn đến thư mục chứa toàn bộ dữ liệu mới thêm vào (ví dụ thư mục dữ liệu mới thêm vào đặt tên là 125 bên trong có toàn bộ các frame được phân bố trong từng folder con tương ứng), argument 3 là đường dẫn đến file images_name và argument 4 là đường dẫn lưu file detection (YoloV3 Human Detection).
  Tiếp đến chạy đoạn lệnh sau để nhận diện xương người
  ![image](https://github.com/user-attachments/assets/38bdc75b-9d6d-4fdb-be10-cc3d12ec0a58)
Nó gồm 4 arguments: argument đầu tiên là đường dẫn đến file prepare_pose_estimation.py (bạn có thể tìm thấy file này bên trong thư mục HRNet), argument thứ hai là đường dẫn đến thư mục chứa toàn bộ dữ liệu mới thêm vào (ví dụ thư mục dữ liệu mới thêm vào đặt tên là 125 bên trong có toàn bộ các frame được phân bố trong từng folder con tương ứng), argument 3 là đường dẫn đến file detectors mới thực hiện và argument 4 là đường dẫn lưu file pose estimation (HR Net Pose Estimation).
* Bước 4:
  ![image](https://github.com/user-attachments/assets/50d1a862-970d-4431-90a8-97e1108add3c)
  Đổi đường dẫn đến file estimation vừa mới tạo và đường dẫn lưu file csv mới.
  Tiếp theo, ta ghép file csv dữ liệu train và dữ liệu pose-estimation mới lại với nhau, tương tự là dữ liệu test và dữ liệu pose-estimation mới. Đổi đường dẫn và tiến hành train lại với dữ liệu mới. Khi đánh giá mô hình mới, có thể đánh giá trên nguyên tập dữ liệu test hoặc là lấy riêng dữ liệu pose-estimation mới để test xem mô hình có nhận dạng được mẫu mới không.






