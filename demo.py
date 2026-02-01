import os
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from net.Flow_DistillModel import FDRect
import tempfile

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RectificationDemo:
    def __init__(self, model_path, device='cuda:0'):
        """
        初始化矩形化演示器

        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('cuda:0' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.resize_w = 512
        self.resize_h = 384

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])

        # 加载模型
        self.model = self.load_model(model_path)

        print(f"模型加载完成，运行在 {self.device}")

    def load_model(self, model_path):
        """加载并初始化模型"""
        # 创建模型
        model = FDRect()

        # 加载预训练权重
        if os.path.exists(model_path):
            pretrain_model = torch.load(model_path, map_location='cpu')
            model_dict = model.state_dict()

            # 过滤出可用的权重
            state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}

            # 更新模型权重
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

            print(f"成功加载模型权重: {model_path}")
        else:
            print(f"警告: 未找到模型文件 {model_path}，使用随机初始化权重")

        # 移动到设备
        model = model.to(self.device)
        model.eval()

        # 打印模型参数数量
        total_params = sum([param.nelement() for param in model.parameters()])
        print(f"模型参数量: {total_params / 1e6:.2f}M")

        return model

    def preprocess_image(self, image):
        """预处理上传的图像"""
        # 确保图像是RGB格式
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA图
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return image

    def preprocess_mask(self, mask):
        """预处理掩码图像"""
        # 如果掩码是None，返回None
        if mask is None:
            return None

        # 确保掩码是合适的格式
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:  # 灰度图
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif mask.shape[2] == 4:  # RGBA图
                mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)

            # 转换为PIL Image
            mask = Image.fromarray(mask)

        return mask

    def create_default_mask(self, image_size):
        """创建默认掩码（全白，表示没有遮挡）"""
        # 创建一个全白的掩码，表示没有需要修复的区域
        mask = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
        return Image.fromarray(mask)

    def rectification_process(self, input_image, mask_image=None):
        """执行矩形化过程"""
        try:
            # 预处理输入图像
            input_pil = self.preprocess_image(input_image)

            # 处理掩码图像
            if mask_image is None:
                # 如果没有提供掩码，创建默认掩码
                mask_pil = self.create_default_mask(input_pil.size)
            else:
                # 预处理掩码图像
                mask_pil = self.preprocess_mask(mask_image)

                # 确保掩码尺寸与输入图像一致
                if mask_pil.size != input_pil.size:
                    mask_pil = mask_pil.resize(input_pil.size, Image.Resampling.NEAREST)

            # 转换为tensor
            with torch.no_grad():
                input_tensor = self.transform(input_pil).unsqueeze(0).float().to(self.device)
                mask_tensor = self.transform(mask_pil).unsqueeze(0).float().to(self.device)

                # 前向传播
                flow, warp_mask_final, final_image = self.model.forward(input_tensor, mask_tensor)

                # 将结果转换回numpy图像
                result = final_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result = np.clip(result, 0, 1)  # 确保值在[0,1]范围内
                result = (result * 255).astype(np.uint8)

                # 转换为PIL图像
                result_pil = Image.fromarray(result)

                return result_pil

        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回错误图像或原始图像
            return input_pil

    def process_images(self, input_image, mask_image=None):
        """
        处理单张或多张图像

        Args:
            input_image: 输入图像（可以是PIL Image或numpy数组）
            mask_image: 自定义掩码图像

        Returns:
            rectified_image: 矩形化后的图像
        """
        try:
            return self.rectification_process(input_image, mask_image)

        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
            return input_image


def create_gradio_interface(model_path):
    """创建Gradio界面"""
    # 初始化演示器
    demo = RectificationDemo(model_path)

    # Gradio界面定义 - 使用新版本的API
    with gr.Blocks(title="FDRect图像矩形化", css="""
        .container {
            max-width: 1400px;
            margin: auto;
            padding: 20px;
        }
        .header-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: white;
            text-align: center;
        }
        .input-section {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #f0f9f0;
        }
        .mask-section {
            border: 2px dashed #FF9800;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #FFF3E0;
        }
        .result-section {
            border: 2px solid #2196F3;
            border-radius: 10px;
            padding: 20px;
            background-color: #E3F2FD;
            margin-top: 20px;
        }
        .example-container {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 5px;
            background: white;
            transition: all 0.3s;
            cursor: pointer;
        }
        .example-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .example-image {
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .example-title {
            font-weight: bold;
            text-align: center;
            margin-bottom: 5px;
            color: #333;
        }
        .example-desc {
            font-size: 12px;
            text-align: center;
            color: #666;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-example {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin-top: 5px;
        }
        .btn-example:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-secondary {
            background: #f5f5f5;
            color: #333;
            border: 1px solid #ddd;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        .example-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
    """) as interface:

        # 顶部标题区域
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                <div class="header-section">
                    <h1 style="margin: 0; font-size: 28px;">🖼️ FDRect图像矩形化</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">上传输入图像和掩码，系统将进行矩形化校正</p>
                </div>
                """)

        # 主要输入区域
        with gr.Row():
            # 输入图像列
            with gr.Column(scale=1):
                with gr.Group():  # 输入图像容器
                    gr.Markdown("### 📤 输入图像")
                    gr.Markdown("上传需要矩形化的图像")

                    # 输入图像上传组件
                    input_image = gr.Image(
                        label="输入图像",
                        type="pil",
                        height=300,
                        interactive=True
                    )

                    # 输入图像上传按钮
                    input_upload_button = gr.UploadButton(
                        "📁 上传输入图像",
                        file_types=["image"],
                        file_count="single",
                        variant="primary",
                        scale=1
                    )

            # 掩码图像列
            with gr.Column(scale=1):
                with gr.Group():  # 掩码图像容器
                    gr.Markdown("### 🎭 掩码图像 (可选)")
                    gr.Markdown("上传掩码图像，黑色区域表示需要修复")

                    # 掩码图像上传组件
                    mask_image = gr.Image(
                        label="掩码图像",
                        type="pil",
                        height=300,
                        interactive=True
                    )

                    # 掩码图像上传按钮
                    mask_upload_button = gr.UploadButton(
                        "📁 上传掩码图像",
                        file_types=["image"],
                        file_count="single",
                        variant="primary",
                        scale=1
                    )

        # 示例图像区域 - 水平布局
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 示例图像")
                gr.Markdown("点击下面的示例图像快速测试")

                # 创建示例图像容器
                with gr.Row():
                    # 检查示例文件夹
                    if os.path.exists("./examples"):
                        # 获取所有图像文件
                        all_files = os.listdir("examples")
                        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

                        if image_files:
                            # 最多显示4个示例
                            for i, img_file in enumerate(image_files[:5]):
                                img_path = os.path.join("examples", img_file)
                                if os.path.exists(img_path):
                                    base_name = os.path.splitext(img_file)[0]

                                    # 检查是否有对应的掩码文件
                                    mask_candidates = [
                                        f for f in image_files
                                        if ('mask' in f.lower() and base_name in f) or
                                           (f.lower().startswith('mask') and base_name.split('_')[-1] in f)
                                    ]

                                    if mask_candidates:
                                        mask_path = os.path.join("examples", mask_candidates[0])
                                        mask_available = True
                                    else:
                                        mask_path = None
                                        mask_available = False

                                    # 创建示例卡片
                                    with gr.Column(min_width=200):
                                        # 显示缩略图
                                        with gr.Group():
                                            # 显示示例图像
                                            example_img = gr.Image(
                                                value=img_path,
                                                label=f"示例{i + 1}",
                                                type="filepath",
                                                height=100,
                                                interactive=False,
                                                show_label=False,
                                                elem_classes="example-image"
                                            )

                                            # 示例标题
                                            gr.Markdown(f"**示例 {i + 1}**", elem_classes="example-title")

                                            # 示例描述
                                            if mask_available:
                                                gr.Markdown("包含掩码", elem_classes="example-desc")
                                            else:
                                                gr.Markdown("无掩码", elem_classes="example-desc")

                                            # 点击加载按钮 - 使用primary样式
                                            load_btn = gr.Button(
                                                f"使用示例 {i + 1}",
                                                size="sm",
                                                variant="primary",  # 改为primary样式
                                                min_width=150,
                                                elem_classes="btn-example"  # 添加自定义样式类
                                            )

                                            # 点击事件
                                            def load_example(img_path, mask_path, i=i):
                                                if mask_path:
                                                    return gr.Image(value=img_path), gr.Image(value=mask_path)
                                                else:
                                                    return gr.Image(value=img_path), None

                                            load_btn.click(
                                                fn=load_example,
                                                inputs=[gr.State(img_path), gr.State(mask_path)],
                                                outputs=[input_image, mask_image]
                                            )
                        else:
                            gr.Markdown("示例文件夹中没有图像文件")
                    else:
                        gr.Markdown("示例文件夹不存在")

        # 控制按钮区域
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    # 处理按钮
                    process_btn = gr.Button(
                        "🚀 开始矩形化处理",
                        variant="primary",
                        size="lg",
                        scale=2
                    )

                    # 清除按钮
                    clear_btn = gr.Button(
                        "🗑️ 清除所有",
                        variant="secondary",
                        scale=1
                    )

        # 结果输出区域
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 📐 矩形化结果")
                    gr.Markdown("处理后的矩形化图像")

                    # 结果图像显示
                    output_image = gr.Image(
                        label="矩形化结果",
                        type="pil",
                        height=400,
                        interactive=False
                    )

        # 底部信息区域
        with gr.Row():
            with gr.Column(scale=2):
                # 处理信息
                info_text = gr.Textbox(
                    label="处理信息",
                    value="等待处理...",
                    interactive=False
                )

            with gr.Column(scale=1):
                # 下载按钮
                download_btn = gr.Button(
                    "💾 下载结果",
                    variant="secondary",
                    size="lg"
                )

        # 处理按钮点击事件
        def process_image(input_img, mask_img):
            if input_img is None:
                return None, "请先上传输入图像"

            try:
                # 处理图像
                result = demo.process_images(input_img, mask_img)

                if result is None:
                    return None, "❌ 处理失败"

                return result, "✅ 处理完成！"

            except Exception as e:
                print(f"处理错误: {e}")
                import traceback
                traceback.print_exc()
                return None, f"❌ 处理失败: {str(e)}"

        # 绑定事件
        input_upload_button.upload(
            lambda x: x,
            inputs=[input_upload_button],
            outputs=[input_image]
        )

        mask_upload_button.upload(
            lambda x: x,
            inputs=[mask_upload_button],
            outputs=[mask_image]
        )

        process_btn.click(
            fn=process_image,
            inputs=[input_image, mask_image],
            outputs=[output_image, info_text]
        )

        # 清除按钮事件
        def clear_all():
            return None, None, None, "已清除所有输入"

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_image, mask_image, output_image, info_text]
        )

        # 下载按钮事件
        def prepare_download_result(result_img):
            if result_img is None:
                return None
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                result_img.save(tmp.name)
                return tmp.name

        download_btn.click(
            fn=prepare_download_result,
            inputs=[output_image],
            outputs=gr.File(label="下载矩形化结果")
        )

        # 添加使用说明
        with gr.Accordion("ℹ️ 详细使用说明", open=False):
            gr.Markdown("""
            ## 🎯 系统功能说明

            本系统用于图像的矩形化校正处理，可以同时提供输入图像和掩码图像：

            ### 1. **输入图像要求**
            - 需要矩形化的扭曲/变形图像
            - 支持格式：JPG、PNG、JPEG
            - 建议尺寸：1024×768 以内以获得最佳性能

            ### 2. **掩码图像要求 (可选)**
            - 黑色区域：需要修复/矩形化的区域
            - 白色区域：原图像的区域
            - 如果不提供掩码，系统会使用全白掩码（处理整个图像）
            - 掩码应与输入图像尺寸一致

            ### 3. **工作流程**
            1. **上传输入图像**：左侧上传需要处理的图像
            2. **上传掩码图像**：右侧上传对应的掩码（可选）
            3. **开始处理**：点击"开始矩形化处理"按钮
            4. **查看结果**：下方显示矩形化结果
            5. **下载结果**：点击下载按钮保存处理结果

            ### 4. **示例使用**
            - 点击上方的"示例图像"快速加载测试用例
            - 示例包含带有掩码和不带掩码的情况
            - 可以直接使用示例图像进行测试

            ### 5. **应用场景**
            - 文档图像矫正
            - 海报/照片矩形化
            - 建筑图像校正
            - 任何需要从透视变形恢复为矩形的图像

            ### 6. **注意事项**
            - 处理时间与图像大小成正比
            - 大图像（>10MB）可能需要较长时间
            - 掩码精度直接影响处理结果
            - 建议使用高对比度的输入图像
            """)

    return interface


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="图像矩形化Gradio演示")
    parser.add_argument("--model_path", type=str, default="../model/distill_model_epoch200.pkl",
                        help="模型权重文件路径")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务器端口")
    parser.add_argument("--share", action="store_true",
                        help="是否创建公网链接")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备 (cuda:0 或 cpu)")

    args = parser.parse_args()

    # 创建Gradio界面
    interface = create_gradio_interface(args.model_path)

    # 启动服务
    print("🚀 启动图像矩形化演示系统...")
    print("=" * 60)
    print(f"📁 模型路径: {args.model_path}")
    print(f"⚙️  计算设备: {args.device}")
    print(f"🌐 服务端口: {args.port}")
    print(f"🔗 公网分享: {args.share}")
    print("=" * 60)
    print("💻 请在浏览器中访问以下地址：")
    print(f"👉 http://127.0.0.1:{args.port}")
    print(f"👉 http://localhost:{args.port}")
    print("=" * 60)
    print("按 Ctrl+C 停止服务器\n")

    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=args.share,
            debug=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n🔧 常见问题解决：")
        print("1. 端口被占用：尝试使用不同的端口，例如 --port 7861")
        print("2. 模型文件不存在：检查 --model_path 参数是否正确")
        print("3. CUDA不可用：尝试使用 --device cpu")
        print("4. 依赖缺失：确保已安装 torch, torchvision, opencv-python, gradio")
        print("5. Gradio版本：请安装较新版本的gradio: pip install --upgrade gradio")


if __name__ == "__main__":
    main()