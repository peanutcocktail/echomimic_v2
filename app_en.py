"""
@界面作者：十字鱼 <https://space.bilibili.com/893892>
"""
import os
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from moviepy.editor import VideoFileClip, AudioFileClip

import gradio as gr
from datetime import datetime
from torchao.quantization import quantize_, int8_weight_only
import gc

total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
print(f'\033[32mCUDA版本：{torch.version.cuda}\033[0m')
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
print(f'\033[32m精度：float16\033[0m')
dtype = torch.float16
if torch.cuda.is_available():
        device = "cuda"
else:
    print("cuda not available, using cpu")
    device = "cpu"

#ffmpeg_path = os.getenv('FFMPEG_PATH')
#if ffmpeg_path is None:
#    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
#elif ffmpeg_path not in os.getenv('PATH'):
#    print("add ffmpeg to path")
#    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def generate(image_input, audio_input, pose_input, width, height, length, steps, sample_rate, cfg, fps, context_frames, context_overlap, quantization_input, seed):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True, parents=True)

    ############# model_init started #############
    ## vae init
    vae = AutoencoderKL.from_pretrained("./pretrained_weights/sd-vae-ft-mse").to(device, dtype=dtype)
    if quantization_input:
        quantize_(vae, int8_weight_only())
        print("使用int8量化")

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained("./pretrained_weights/sd-image-variations-diffusers", subfolder="unet").to(dtype=dtype, device=device)
    reference_unet.load_state_dict(torch.load("./pretrained_weights/reference_unet.pth", weights_only=True))
    if quantization_input:
        quantize_(reference_unet, int8_weight_only())

    ## denoising net init
    if os.path.exists("./pretrained_weights/motion_module.pth"):
        print('using motion module')
    else:
        exit("motion module not found")
        ### stage1 + stage2
    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        "./pretrained_weights/sd-image-variations-diffusers",
        "./pretrained_weights/motion_module.pth",
        subfolder="unet",
        unet_additional_kwargs = {
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "cross_attention_dim": 384,
            "motion_module_resolutions": [
                1,
                2,
                4,
                8
            ],
            "motion_module_mid_block": True ,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs":{
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": [
                    'Temporal_Self',
                    'Temporal_Self'
                ],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            }
        },
    ).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(torch.load("./pretrained_weights/denoising_unet.pth", weights_only=True),strict=False)

    # pose net init
    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
    pose_net.load_state_dict(torch.load("./pretrained_weights/pose_encoder.pth", weights_only=True))

    ### load audio processor params
    audio_processor = load_audio_model(model_path="./pretrained_weights/audio_processor/tiny.pt", device=device)
   
    ############# model_init finished #############
    sched_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "clip_sample": False,
        "steps_offset": 1,
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing"
    }
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=dtype)

    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        seed = random.randint(100, 1000000)
        generator = torch.manual_seed(seed)

    inputs_dict = {
        "refimg": image_input,
        "audio": audio_input,
        "pose": pose_input,
    }

    print('Pose:', inputs_dict['pose'])
    print('Reference:', inputs_dict['refimg'])
    print('Audio:', inputs_dict['audio'])

    save_name = f"{save_dir}/{timestamp}"
    
    ref_image_pil = Image.open(inputs_dict['refimg']).resize((width, height))
    audio_clip = AudioFileClip(inputs_dict['audio'])
    
    length = min(length, int(audio_clip.duration * fps), len(os.listdir(inputs_dict['pose'])))

    start_idx = 0

    pose_list = []
    for index in range(start_idx, start_idx + length):
        tgt_musk = np.zeros((width, height, 3)).astype('uint8')
        tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(index))
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im),(1, 2, 0))
        tgt_musk[rb:re,cb:ce,:] = im

        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
        pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=dtype, device=device).permute(2,0,1) / 255.0)
    
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    audio_clip = AudioFileClip(inputs_dict['audio'])
    
    audio_clip = audio_clip.set_duration(length / fps)
    video = pipe(
        ref_image_pil,
        inputs_dict['audio'],
        poses_tensor[:,:,:length,...],
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap,
        start_idx=start_idx,
    ).videos 
    
    final_length = min(video.shape[2], poses_tensor.shape[2], length)
    video_sig = video[:, :, :final_length, :, :]
    
    save_videos_grid(
        video_sig,
        save_name + "_woa_sig.mp4",
        n_rows=1,
        fps=fps,
    )

    video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4",)
    video_clip_sig = video_clip_sig.set_audio(audio_clip)
    video_clip_sig.write_videofile(save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2)
    video_output = save_name + "_sig.mp4"
    seed_text = gr.update(visible=True, value=seed)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return video_output, seed_text


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">EchoMimicV2</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/antgroup/echomimic_v2">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2411.10061">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ This demo is for academic research and experiential use only.
            </div>
            
            """)
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image_input = gr.Image(label="Input Image", type="filepath")
                    audio_input = gr.Audio(label="Input Audio", type="filepath")
                    pose_input = gr.Textbox(label="Input Pose (Directory Path)", placeholder="Please enter the directory path for pose data", value="assets/halfbody_demo/pose/01")
                with gr.Group():
                    with gr.Row():
                        width = gr.Number(label="width", value=768)
                        height = gr.Number(label="height", value=768)
                        length = gr.Number(label="length", value=240)
                    with gr.Row():
                        steps = gr.Number(label="steps", value=20)
                        sample_rate = gr.Number(label="sample rate", value=16000)
                        cfg = gr.Number(label="cfg", value=2.5, step=0.1)
                    with gr.Row():
                        fps = gr.Number(label="fps", value=24)
                        context_frames = gr.Number(label="context frames", value=12)
                        context_overlap = gr.Number(label="context overlap", value=3)
                    with gr.Row():
                        quantization_input = gr.Checkbox(label="INT8 Quantization (Recommended for users with 12GB VRAM, and for audio no longer than 5 seconds)", value=False)
                        seed = gr.Number(label="seed (-1 for random)", value=-1)
                generate_button = gr.Button("🎬 Generate")
            with gr.Column():
                video_output = gr.Video(label="Output")
                seed_text = gr.Textbox(label="seed", interactive=False, visible=False)
        gr.Examples(
            examples=[
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0003.png", "assets/halfbody_demo/audio/chinese/fighting.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png", "assets/halfbody_demo/audio/chinese/good.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0010.png", "assets/halfbody_demo/audio/chinese/news.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/1168.png", "assets/halfbody_demo/audio/chinese/no_smoking.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png", "assets/halfbody_demo/audio/chinese/ultraman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0001.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0003.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0010.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0017.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0025.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0055.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0056.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0101.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0119.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0154.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0170.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0177.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0181.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0211.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0252.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0324.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0398.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0415.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0424.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/1168.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0010.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0035.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0048.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0077.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0101.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0140.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0163.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0175.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0201.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0212.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0215.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0247.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0253.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0269.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0284.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0287.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0430.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0588.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
            ],
            inputs=[image_input, audio_input],  
            label="Preset Characters and Audio",
        )

    generate_button.click(
        generate,
        inputs=[image_input, audio_input, pose_input, width, height, length, steps, sample_rate, cfg, fps, context_frames, context_overlap, quantization_input, seed],
        outputs=[video_output, seed_text],
    )



if __name__ == "__main__":
    demo.queue()
    demo.launch()
#    demo.launch(inbrowser=True)
