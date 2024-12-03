## Cài đặt môi trường
* Môi trường:
  * **Python**, **pip**
  * **TesseractOCR** và dependencies
    * `sudo apt install tesseract-ocr `
    * `sudo apt install ghostscript`
    * `sudo apt-get install tesseract-ocr-vie`
  * **OCRmyPDF**
    * `pip install -e .`

Những câu lệnh này được ghi trong `setup.sh`
  

## Getting Started
Câu lệnh mẫu: `ocrmypdf input.pdf output.pdf -l eng+vie`

Trong đó:
* `input.pdf`: file đầu vào 
* `output.pdf`: file đầu ra
*  `-l eng+vie`: Hiện tại hỗ trợ 2 ngôn ngữ ENG và VIE


## Workflow
![OCRmyPDF drawio (2)](https://github.com/user-attachments/assets/13af5b28-83df-4ffc-a2a6-52d46c6a28f5)

Đầu vào là file PDF. File này được tiền xử lý và trích xuất nội dung văn bản bằng **TesseractOCR**. Tiếp theo, **LLM Component** đảm nhận việc sửa lỗi chính tả và ngữ pháp của nội dung, đồng thời tạo ra một file tóm tắt. Nội dung sau khi được chỉnh sửa sẽ được layer lại vào file PDF gốc.

Kết quả đầu ra của công cụ bao gồm ba file:

* File nội dung đã được sửa lỗi.
* File tóm tắt nội dung.
* File PDF với nội dung đã được sửa lỗi.

## Cấu trúc mã nguồn OCRmyPDF
Cấu trúc mã nguồn của công cụ gồm các folder:
* `_exec`: Thực thi những chức năng cốt lõi của công cụ, ví dụ `tesseract.py` xử lý những tác vụ OCR PDF.
* `_pipelines`: Xử lý chức năng tầng cao hơn, liên quan đến luồng hoạt động của công cụ như (PDF to hOCR)
* `builtin_plugins`: Các thư viện core mà **OCRmyPDF** sử dụng gọi là plugin, và  **Tesseract** là một trong những plugin mặc định (builtin) 
* `data`
* `extra_plugins`: Các plugin mở rộng mà người dùng có thể tích hợp thêm. Ví dụ như thêm thư viện OCR khác (**EasyOCR**)
* `hocrtransform`: Thực thi chức năng layer PDF với nội dung được OCR bởi Tesseract
* `llm_text_improve`: Thực thi chức năng sửa lỗi chính tả và tóm tắt nội dung bằng cách ứng dụng **Large Language Model**. Hiện tại, công cụ đang tích hợp model [PhoGPT-4B](https://huggingface.co/vinai/PhoGPT-4B).
* `pdfinfo`: Thực thi chức năng tiền xử lý PDF.
* `subprocess`

## Debug

Để hiểu luồng hoạt động của công cụ, ta bắt đầu từ việc thực thi test case sau:
``` python
def test_simple_input_pdf(resources, outpdf):
    result = run_ocrmypdf_api("input.pdf", "output.pdf","-l","eng+vie")
    assert result == ExitCode.ok
```

Trong test case này hàm `run_ocrmypdf_api` là hàm khởi đầu của luồng hoạt động của **OCRmyPDF**

Bên trong hàm `run_ocrmypdf_api`, hàm `run_pipeline_cli` đóng vai trò quan trọng, nhận đầu vào là các thành phần của 1 câu lệnh OCRmyPDF đã được parsed, đảm nhận xử lý các câu lệnh CLI.
```python
def run_ocrmypdf_api(input_file: Path, output_file: Path, *args) -> ExitCode:
    """Run ocrmypdf via its API in-process, but return CLI-style ExitCode.

    This simulates calling the command line interface in a subprocess and allows us
    to check that the command line interface is working correctly, but since it is
    in-process it is easier to trace with a debugger or coverage tool.

    Any exception raised will be trapped and converted to an exit code.
    The return code must be checked by the caller to determine if the test passed.
    """
    api_args = [str(input_file), str(output_file)] + [
        str(arg) for arg in args if arg is not None
    ]
    _parser, options, plugin_manager = get_parser_options_plugins(args=api_args)

    api.check_options(options, plugin_manager)
    return api.run_pipeline_cli(options, plugin_manager=plugin_manager)
```

Ở tầng logic sâu hơn, hàm `_run_pipeline` trực tiếp xử lý logic của công cụ, cần đặc biệt quan tâm.
```python
def _run_pipeline(
    options: argparse.Namespace,
    plugin_manager: OcrmypdfPluginManager,
) -> ExitCode:
```





